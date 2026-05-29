"""FinanceBench benchmark builder.

Financial QA over SEC filings. ``download`` fetches the 150-example dataset and
the referenced filing PDFs (into ``spec.source_dir``); ``build_corpus`` extracts
text page-by-page from those PDFs as the retrieval corpus.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from marlib.benchmarks.base import BenchmarkBuilder, BenchmarkSpec, register, slugify
from marlib.log import logger

# Official SEC filing PDFs referenced by the dataset.
_PDF_BASE_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"


@register("financebench")
class FinanceBenchBuilder(BenchmarkBuilder):
    """Builder for the FinanceBench benchmark."""

    def download(self, spec: BenchmarkSpec) -> None:
        """Download the FinanceBench questions and the referenced filing PDFs.

        Args:
            spec: Benchmark spec. Questions -> ``spec.questions_path``,
                PDFs -> ``spec.source_dir``.
        """
        from datasets import load_dataset

        spec.questions_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading FinanceBench dataset...")
        ds = load_dataset("PatronusAI/financebench", split="train")
        logger.info(f"Total examples: {len(ds)}")

        processed = [
            {
                "id": ex["financebench_id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "evidence": ex["evidence"],
                "company": ex.get("company"),
                "doc_name": ex.get("doc_name"),
                "question_type": ex.get("question_type"),
                "question_reasoning": ex.get("question_reasoning"),
                "justification": ex.get("justification"),
                "gics_sector": ex.get("gics_sector"),
                "doc_type": ex.get("doc_type"),
                "doc_period": ex.get("doc_period"),
            }
            for ex in ds
        ]

        logger.info(f"Saving {len(processed)} questions to: {spec.questions_path}")
        with open(spec.questions_path, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")

        self._download_pdfs(spec)
        logger.success("FinanceBench download complete!")

    def _download_pdfs(self, spec: BenchmarkSpec) -> None:
        """Download every filing PDF referenced in the question file."""
        spec.source_dir.mkdir(parents=True, exist_ok=True)

        doc_names: set[str] = set()
        with open(spec.questions_path) as f:
            for line in f:
                doc_name = json.loads(line).get("doc_name")
                if doc_name:
                    doc_names.add(doc_name)
        logger.info(f"Found {len(doc_names)} unique documents to download")

        from tqdm import tqdm

        github_token = os.environ.get("GITHUB_TOKEN")
        downloaded = skipped = failed = 0
        for doc_name in tqdm(sorted(doc_names), desc="Downloading PDFs"):
            pdf_path = spec.source_dir / f"{doc_name}.pdf"
            if pdf_path.exists():
                skipped += 1
                continue
            url = f"{_PDF_BASE_URL}/{urllib.parse.quote(f'{doc_name}.pdf')}"
            try:
                req = urllib.request.Request(url)
                if github_token:
                    req.add_header("Authorization", f"token {github_token}")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    pdf_path.write_bytes(resp.read())
                downloaded += 1
            except Exception as e:
                logger.warning(f"Failed to download {doc_name}: {e}")
                failed += 1
        logger.success(
            f"PDFs: {downloaded} downloaded, {skipped} skipped, {failed} failed"
        )

    def build_corpus(
        self,
        spec: BenchmarkSpec,
        max_paragraphs: int | None = None,
    ) -> None:
        """Build the corpus by extracting text from every page of each filing PDF.

        Args:
            spec: Benchmark spec (PDFs read from ``spec.source_dir``,
                corpus written to ``spec.corpus_path``).
            max_paragraphs: Optional cap on number of pages (for testing).
        """
        import pymupdf
        from tqdm import tqdm

        spec.corpus_path.parent.mkdir(parents=True, exist_ok=True)

        # Metadata enrichment from the question file: doc_name -> metadata.
        metadata_map: dict[str, dict] = {}
        if spec.questions_path.exists():
            logger.info(f"Loading metadata from: {spec.questions_path}")
            with open(spec.questions_path) as f:
                for line in f:
                    entry = json.loads(line)
                    doc_name = entry.get("doc_name")
                    if doc_name and doc_name not in metadata_map:
                        metadata_map[doc_name] = {
                            "company": entry.get("company"),
                            "doc_type": entry.get("doc_type"),
                            "doc_period": entry.get("doc_period"),
                            "gics_sector": entry.get("gics_sector"),
                        }
            logger.info(f"Loaded metadata for {len(metadata_map)} documents")

        pdf_files = sorted(spec.source_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {spec.source_dir}")
            return
        logger.info(f"Processing {len(pdf_files)} PDFs from: {spec.source_dir}")

        seen_ids: set[str] = set()
        paragraphs: list[dict] = []
        skipped_encrypted = skipped_empty = 0

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            doc_name = pdf_path.stem
            try:
                doc = pymupdf.open(pdf_path)
            except Exception as e:
                logger.warning(f"Could not open {pdf_path.name}: {e}")
                continue
            if doc.is_encrypted:
                logger.warning(f"Skipping encrypted PDF: {pdf_path.name}")
                skipped_encrypted += 1
                doc.close()
                continue

            meta = metadata_map.get(doc_name, {})
            for page_idx in range(len(doc)):
                text = doc[page_idx].get_text("text")
                if len(text.strip()) < 50:  # cover/image-only pages
                    skipped_empty += 1
                    continue
                page_number = page_idx + 1  # 1-based, matches evidence_page_num
                doc_id = f"{slugify(doc_name)}_p{page_number}"
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                entry = {
                    "doc_id": doc_id,
                    "title": f"{doc_name} (p. {page_number})",
                    "text": text,
                }
                for key in ("company", "doc_type", "doc_period", "gics_sector"):
                    if meta.get(key):
                        entry[key] = meta[key]
                paragraphs.append(entry)

                if max_paragraphs and len(paragraphs) >= max_paragraphs:
                    break
            doc.close()
            if max_paragraphs and len(paragraphs) >= max_paragraphs:
                break

        logger.info(f"Collected {len(paragraphs)} pages from {len(pdf_files)} PDFs")
        if skipped_encrypted:
            logger.info(f"  Skipped {skipped_encrypted} encrypted PDFs")
        if skipped_empty:
            logger.info(f"  Skipped {skipped_empty} near-empty pages")

        logger.info(f"Saving corpus to: {spec.corpus_path}")
        with open(spec.corpus_path, "w") as f:
            for para in tqdm(paragraphs, desc="Writing"):
                f.write(json.dumps(para) + "\n")
        logger.success("FinanceBench corpus preparation complete!")
