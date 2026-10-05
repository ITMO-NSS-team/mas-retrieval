"""FRAMES recipe: pinned questions, hash-verified frozen HTML, shared corpus.

The snapshot manifest must be made from an independently obtained frozen corpus;
this builder never scrapes live Wikipedia or subsets the corpus by question IDs.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from urllib.request import urlopen

from marlib.benchmarks.base import BenchmarkBuilder, BenchmarkSpec, register

REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
QUESTIONS_URL = f"https://huggingface.co/datasets/google/frames-benchmark/resolve/{REVISION}/test.tsv"
QUESTIONS_SHA256 = "4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff"
CHUNK_CHARS = 2400
PARSER_VERSION = "frames_html_v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows):
    # An interrupted build must not leave a corpus that preparation will skip.
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.hostname not in {"en.wikipedia.org", "en.m.wikipedia.org"} or not parsed.path.startswith("/wiki/"):
        raise ValueError(f"Unsupported Wikipedia URL: {url!r}")
    title = unicodedata.normalize("NFC", unquote(parsed.path[len('/wiki/'):])).replace(" ", "_")
    if not title:
        raise ValueError("Empty article title")
    return "https://en.wikipedia.org/wiki/" + quote(title, safe="_(),!'/:;@&=+$-.")


def extract_urls(value) -> list[str]:
    if isinstance(value, list):
        return [u for item in value for u in extract_urls(item)]
    value = str(value or "").strip()
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return extract_urls(parsed)
        except (ValueError, SyntaxError):
            pass
    # Commas inside article titles remain; list separators and Markdown wrappers do not.
    urls = re.findall(r"https?://en\.(?:m\.)?wikipedia\.org/wiki/(?:[^\s\],]|,(?=[^\s\],]))+", value)
    cleaned = []
    for url in urls:
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        cleaned.append(url)
    return cleaned


def article_id(url: str) -> str:
    return "wiki-" + digest(normalize_url(url).encode())[:20]


def convert_questions(data: bytes) -> list[dict]:
    rows = []
    for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter="\t"):
        raw_urls = list(dict.fromkeys(u for key, value in row.items()
                                     if key == "wiki_links" or key.startswith("wikipedia_link_")
                                     for u in extract_urls(value)))
        urls = sorted({normalize_url(u) for u in raw_urls})
        if not urls or not row.get("Prompt") or not row.get("Answer"):
            raise ValueError(f"Incomplete question: {row.get('')}")
        source_id = row.get("")
        if source_id is None or not source_id.isdigit():
            raise ValueError("Expected stable integer source ID in first TSV column")
        rows.append({"id": f"frames_{int(source_id):04d}", "question": row["Prompt"],
                     "answer": row["Answer"], "source_id": source_id,
                     "source_urls": raw_urls, "normalized_urls": urls,
                     "gold_doc_ids": sorted({article_id(u) for u in urls}),
                     "reasoning_types": [v.strip() for v in row.get("reasoning_types", "").split("|") if v.strip()],
                     "n_sources": len(urls)})
    if len({q["id"] for q in rows}) != len(rows):
        raise ValueError("Duplicate FRAMES IDs")
    return rows


def split_ids(questions: list[dict], seed: int, pilot_n: int, final_n: int) -> dict:
    if pilot_n < 0 or final_n <= 0 or pilot_n + final_n > len(questions):
        raise ValueError("Invalid pilot/final sample sizes")
    ids = sorted(q["id"] for q in questions)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs")
    random.Random(seed).shuffle(ids)
    return {"seed": seed, "pilot_ids": ids[:pilot_n],
            "final_ids": ids[pilot_n:pilot_n + final_n],
            "unused_ids": ids[pilot_n + final_n:]}


def parse_html(data: bytes, fallback_url: str) -> tuple[str, str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "lxml")
    canonical = soup.find("link", rel="canonical")
    url = normalize_url(canonical.get("href", fallback_url) if canonical else fallback_url)
    title = soup.title.get_text(" ", strip=True) if soup.title else unquote(url.rsplit("/", 1)[-1])
    # Protection badges can use mw-parser-output before the actual article.
    content = soup.find(id="mw-content-text")
    # Category-page lists can be siblings of mw-parser-output within content.
    main = content if content else (soup.body or soup)
    for element in main.select("script, style, nav, .mw-editsection"):
        element.decompose()
    lines = []
    for element in main.find_all(["p", "li", "table", "h1", "h2", "h3", "h4"]):
        if element.find_parent(["table", "li", "p"]):
            continue
        if element.name == "table":
            header = ""
            for row in element.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"], recursive=False)]
                if cells:
                    row_text = " | ".join(cells)
                    if row.find("th") and not row.find("td"):
                        header = row_text
                        lines.append(row_text)
                    else:
                        lines.append((f"[{header}] " if header else "") + row_text)
        else:
            lines.append(element.get_text(" ", strip=True))
    text = "\n".join(line for line in lines if line.strip())
    if not text:
        raise ValueError(f"No article text: {fallback_url}")
    return url, title, text


def chunks(text: str) -> list[str]:
    # Preserve row/list boundaries where possible. Oversized rows repeat their
    # beginning so the row label remains visible after a split.
    pieces = []
    for line in text.splitlines():
        while len(line) > CHUNK_CHARS:
            cut = line.rfind(" ", 0, CHUNK_CHARS)
            cut = cut if cut > 200 else CHUNK_CHARS
            pieces.append(line[:cut])
            line = line[:120] + " … " + line[cut:].lstrip()
        if line:
            pieces.append(line)
    output, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > CHUNK_CHARS:
            output.append(current)
            current = ""
        current += ("\n" if current else "") + piece
    if current:
        output.append(current)
    return output


def freeze_snapshot(source_dir: Path, mapping_path: Path, source_url: str, revision: str):
    """Record an already extracted frozen corpus; no live URL fetching.

    mapping_path is a JSON object {relative HTML filename: original URL}; names
    are relative to source_dir/doc_html, with optional .html extensions.
    """
    mapping = json.loads(mapping_path.read_text())
    if not isinstance(mapping, dict) or not source_url or not revision:
        raise ValueError("Provide filename-to-URL mapping, source URL and revision")
    articles = []
    for path in sorted((source_dir / "doc_html").rglob("*.html")):
        rel = path.relative_to(source_dir / "doc_html").as_posix()
        url = mapping.get(rel, mapping.get(str(Path(rel).with_suffix(""))))
        if not isinstance(url, str):
            raise ValueError(f"Missing source URL mapping for {rel}")
        articles.append({"path": path.relative_to(source_dir).as_posix(),
                         "url": url, "sha256": digest(path.read_bytes())})
    if not articles:
        raise ValueError("No frozen HTML files in source/doc_html")
    write_json(source_dir / "snapshot.json", {"source_url": source_url, "revision": revision,
                                              "articles": articles})


@register("frames")
class FramesBuilder(BenchmarkBuilder):
    def download(self, spec: BenchmarkSpec) -> None:
        spec.source_dir.mkdir(parents=True, exist_ok=True)
        path = spec.source_dir / "test.tsv"
        data = path.read_bytes() if path.exists() else urlopen(QUESTIONS_URL, timeout=60).read()
        if digest(data) != QUESTIONS_SHA256:
            raise ValueError("FRAMES TSV checksum mismatch; do not silently change revisions")
        questions = convert_questions(data)
        if len(questions) != 824:
            raise ValueError("Unexpected question count")
        path.write_bytes(data)
        write_jsonl(spec.questions_path, questions)
        write_json(spec.source_dir / "questions_provenance.json",
                   {"url": QUESTIONS_URL, "revision": REVISION, "sha256": digest(data), "n": len(questions)})

    def build_corpus(self, spec: BenchmarkSpec, max_paragraphs: int | None = None) -> None:
        if max_paragraphs is not None:
            raise ValueError("FRAMES requires the full fixed corpus; use question ID subsets instead")
        manifest_path = spec.source_dir / "snapshot.json"
        if not manifest_path.exists():
            raise FileNotFoundError("FRAMES needs source/doc_html plus source/snapshot.json; see README frozen-corpus preparation")
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("source_url") or not manifest.get("revision") or not manifest.get("articles"):
            raise ValueError("Snapshot provenance is incomplete")
        aliases: dict[str, set[str]] = {}
        documents, sources, seen = [], [], set()
        invalid_documents, conflicting_articles = [], []
        for entry in manifest["articles"]:
            path = (spec.source_dir / entry["path"]).resolve()
            if not path.is_relative_to(spec.source_dir.resolve()):
                raise ValueError("Snapshot file outside source directory")
            data = path.read_bytes()
            if digest(data) != entry["sha256"]:
                raise ValueError(f"HTML checksum mismatch: {entry['path']}")
            try:
                canonical, title, text = parse_html(data, entry["url"])
            except ValueError as exc:
                invalid_documents.append({"path": entry["path"], "url": entry["url"], "error": str(exc)})
                continue
            aid = article_id(canonical)
            for url in [entry["url"], canonical, *entry.get("aliases", [])]:
                aliases.setdefault(normalize_url(url), set()).add(aid)
            # Duplicate redirect copies must agree; never choose silently.
            identity = (aid, digest(text.encode()))
            if any(old[0] == aid and old != identity for old in seen):
                conflicting_articles.append({"url": canonical, "path": entry["path"]})
                continue
            if identity in seen:
                continue
            seen.add(identity)
            chunk_ids = []
            for i, text_chunk in enumerate(chunks(text)):
                cid = f"{aid}_{i:05d}"
                chunk_ids.append(cid)
                documents.append({"doc_id": cid, "article_id": aid, "source_url": canonical,
                                  "title": title, "text": text_chunk})
            sources.append({"article_id": aid, "canonical_url": canonical,
                            "html_sha256": entry["sha256"], "chunk_ids": chunk_ids})
        questions = spec.load_questions()
        missing, ambiguous = [], []
        for q in questions:
            gold = []
            for url in q["normalized_urls"]:
                matches = aliases.get(url, set())
                if not matches:
                    missing.append({"question_id": q["id"], "url": url})
                elif len(matches) > 1:
                    ambiguous.append({"question_id": q["id"], "url": url, "article_ids": sorted(matches)})
                else:
                    gold.extend(matches)
            q["gold_doc_ids"] = sorted(set(gold))
            q["n_corpus_sources"] = len(q["gold_doc_ids"])
        coverage = {"questions": len(questions), "articles": len(sources), "chunks": len(documents),
                    "missing": missing, "ambiguous": ambiguous,
                    "invalid_documents": invalid_documents, "conflicting_articles": conflicting_articles,
                    "complete": not any((missing, ambiguous, invalid_documents, conflicting_articles))}
        write_json(spec.root / "coverage.json", coverage)
        if not coverage["complete"]:
            raise ValueError("FRAMES coverage incomplete; see coverage.json. No questions were excluded")
        write_jsonl(spec.corpus_path, documents)
        write_jsonl(spec.questions_path, questions)
        write_json(spec.root / "sources.json", {"articles": sources,
                   "aliases": {k: sorted(v) for k, v in aliases.items()}})
        import bs4
        from importlib.metadata import version
        write_json(spec.root / "corpus_provenance.json", {
            "snapshot_sha256": digest(manifest_path.read_bytes()),
            "source_url": manifest["source_url"], "revision": manifest["revision"],
            "parser": PARSER_VERSION, "beautifulsoup4": bs4.__version__, "lxml": version("lxml"), "python": sys.version,
            "chunk_chars": CHUNK_CHARS, "overlap": "none; long row prefix 120 chars",
            "corpus_sha256": digest(spec.corpus_path.read_bytes()),
            "questions_sha256": digest(spec.questions_path.read_bytes()), **coverage})
