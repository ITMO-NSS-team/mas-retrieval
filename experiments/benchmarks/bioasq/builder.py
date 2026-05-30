"""BioASQ Task B benchmark builder.

Biomedical QA over PubMed. ``download`` filters the raw BioASQ Task B questions
to the answer-extractive types (factoid/list), draws a *stratified* random sample
(one quota per type, like the HotpotQA builder does for bridge/comparison), and
records the gold PubMed IDs as ``gold_doc_ids``. ``build_corpus`` fetches the
abstracts of those gold PubMed IDs via NCBI E-utilities and writes them as the
retrieval corpus, so ``doc_id == PubMed ID`` and ``context_recall`` matches exactly.

Data access note
-----------------
BioASQ datasets are gated behind free registration at http://bioasq.org. There is
no `datasets.load_dataset` path, so this builder reads the raw JSON you downloaded
(``{"questions": [...]}``) from the benchmark's ``source/`` dir. Drop the file
(e.g. ``training13b.json`` or a ``*_golden.json`` test batch) there before running
``just prepare bioasq``.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from marlib.benchmarks.base import BenchmarkBuilder, BenchmarkSpec, register
from marlib.log import logger

# Answer-extractive types only: their gold answers are short entities/lists, so
# exact_match / f1 are meaningful. yesno (binary) and summary (long ideal_answer)
# would need a different metric setup, so they are excluded by default.
QUESTION_TYPES: tuple[str, ...] = ("factoid", "list")

# Total questions to sample, split evenly across QUESTION_TYPES (cf. HotpotQA=500).
SAMPLE_N = 500
SEED = 42

# NCBI E-utilities (no API key required, but rate-limited to ~3 req/s; set
# NCBI_API_KEY to raise it to 10/s and NCBI_EMAIL to identify yourself politely).
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_PMID_BATCH = 200
_PMID_RE = re.compile(r"(\d+)\s*$")


def _read_raw_questions(spec: BenchmarkSpec) -> list[dict]:
    """Load raw BioASQ Task B questions from any JSON file in ``source/``.

    Returns an empty list (rather than raising) when no source file is present,
    so a bare ``just prepare`` (= prepare *all*) skips this gated benchmark
    instead of aborting the whole sweep.
    """
    src = spec.source_dir
    files = sorted(src.glob("*.json")) if src.is_dir() else []
    if not files:
        return []
    questions: list[dict] = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        questions.extend(data.get("questions", []))
    logger.info(f"Loaded {len(questions)} raw questions from {len(files)} file(s)")
    return questions


def _pmid(url_or_id: str) -> str | None:
    """Extract the trailing PubMed numeric id from a document URL (or plain id)."""
    m = _PMID_RE.search(url_or_id.strip())
    return m.group(1) if m else None


def _normalize_answer(qtype: str, exact: object) -> tuple[str, list[str]]:
    """Reduce a BioASQ ``exact_answer`` to (primary_answer, aliases).

    Formats in the wild:
      factoid -> ["ans", "synonym", ...]  or  [["ans", "synonym", ...]]
      list    -> [["a1", "syn"], ["a2"], ...]   (each entry = one item + synonyms)
    """
    if qtype == "list" and isinstance(exact, list):
        items = [grp[0] if isinstance(grp, list) and grp else grp for grp in exact]
        items = [str(i) for i in items if i]
        return "; ".join(items), items
    if isinstance(exact, list):
        flat = exact[0] if exact and isinstance(exact[0], list) else exact
        aliases = [str(a) for a in flat] if isinstance(flat, list) else [str(flat)]
        return (aliases[0] if aliases else ""), aliases
    return (str(exact) if exact is not None else ""), []


@register("bioasq")
class BioASQBuilder(BenchmarkBuilder):
    """Builder for the BioASQ Task B biomedical QA benchmark."""

    def download(
        self,
        spec: BenchmarkSpec,
        sample_n: int = SAMPLE_N,
        seed: int = SEED,
        types: tuple[str, ...] = QUESTION_TYPES,
    ) -> None:
        """Filter to ``types`` and write a per-type stratified question sample.

        Args:
            spec: Benchmark spec (questions written to ``spec.questions_path``).
            sample_n: Total questions, split evenly across ``types``.
            seed: Random seed for reproducibility.
            types: BioASQ question types to keep (default: factoid + list).
        """
        spec.questions_path.parent.mkdir(parents=True, exist_ok=True)
        raw = _read_raw_questions(spec)
        if not raw:
            logger.warning(
                f"bioasq: no Task B JSON in {spec.source_dir} — skipping. Register "
                "at http://bioasq.org and drop a Task B file (training or *_golden "
                "batch) there, then re-run `just prepare bioasq`."
            )
            return

        by_type: dict[str, list[dict]] = {t: [] for t in types}
        for q in raw:
            if q.get("type") in by_type:
                by_type[q["type"]].append(q)
        for t, qs in by_type.items():
            logger.info(f"{t}: {len(qs)} available")

        random.seed(seed)
        n_per_type = sample_n // len(types)
        sampled: list[dict] = []
        for t in types:
            pool = by_type[t]
            sampled += random.sample(pool, min(n_per_type, len(pool)))
        random.shuffle(sampled)

        processed: list[dict] = []
        for q in sampled:
            answer, aliases = _normalize_answer(q["type"], q.get("exact_answer"))
            gold_doc_ids = sorted(
                {pid for d in q.get("documents", []) if (pid := _pmid(d))}
            )
            processed.append(
                {
                    "id": q["id"],
                    "question": q["body"],
                    "answer": answer,
                    "answer_aliases": aliases,
                    "type": q["type"],
                    # Gold relevance in corpus id space (doc_id == PubMed id).
                    "gold_doc_ids": gold_doc_ids,
                }
            )

        logger.info(f"Saving {len(processed)} questions to: {spec.questions_path}")
        with open(spec.questions_path, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        logger.success("BioASQ download complete!")

    def build_corpus(
        self,
        spec: BenchmarkSpec,
        max_paragraphs: int | None = None,
    ) -> None:
        """Fetch the gold PubMed abstracts as the retrieval corpus.

        Builds the corpus from the union of ``gold_doc_ids`` across the sampled
        questions (optionally capped by ``max_paragraphs``).

        Args:
            spec: Benchmark spec (corpus written to ``spec.corpus_path``).
            max_paragraphs: Optional cap on number of abstracts (for testing).
        """
        if not spec.questions_path.exists():
            logger.warning(
                "bioasq: no questions.jsonl (download skipped — needs Task B data). "
                "Skipping corpus."
            )
            return
        spec.corpus_path.parent.mkdir(parents=True, exist_ok=True)

        with open(spec.questions_path) as f:
            questions = [json.loads(line) for line in f]
        pmids = sorted({pid for q in questions for pid in q.get("gold_doc_ids", [])})
        if max_paragraphs:
            pmids = pmids[:max_paragraphs]
        logger.info(f"Fetching {len(pmids)} unique PubMed abstracts...")

        docs = self._fetch_abstracts(pmids)

        logger.info(f"Saving {len(docs)} abstracts to: {spec.corpus_path}")
        with open(spec.corpus_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
        logger.success("BioASQ corpus preparation complete!")

    @staticmethod
    def _fetch_abstracts(pmids: list[str]) -> list[dict]:
        """Fetch title+abstract for each PubMed id via E-utilities efetch (XML)."""
        import os

        from tqdm import tqdm

        api_key = os.environ.get("NCBI_API_KEY")
        email = os.environ.get("NCBI_EMAIL")
        delay = 0.11 if api_key else 0.34  # respect 10/s (key) vs 3/s (anon)

        docs: list[dict] = []
        for start in tqdm(range(0, len(pmids), _PMID_BATCH), desc="efetch"):
            batch = pmids[start : start + _PMID_BATCH]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "abstract",
                "retmode": "xml",
            }
            if api_key:
                params["api_key"] = api_key
            if email:
                params["email"] = email
            url = f"{_EFETCH_URL}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=60) as resp:
                root = ET.fromstring(resp.read())
            for art in root.findall(".//PubmedArticle"):
                pmid_el = art.find(".//MedlineCitation/PMID")
                if pmid_el is None or not pmid_el.text:
                    continue
                title = "".join(art.find(".//ArticleTitle").itertext()) \
                    if art.find(".//ArticleTitle") is not None else ""
                abstract = " ".join(
                    "".join(node.itertext()).strip()
                    for node in art.findall(".//Abstract/AbstractText")
                )
                text = f"{title} {abstract}".strip()
                if text:
                    docs.append(
                        {"doc_id": pmid_el.text, "title": title, "text": text}
                    )
            time.sleep(delay)
        return docs
