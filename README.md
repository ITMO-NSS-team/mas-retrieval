## Zero-Shot Multi-Agent Generation for Specialized RAG Workflows: An Empirical Evaluation

This repo contains code for preparing data and evaluating auto-generated multi-agent systems on retrieval tasks.
Before running, ensure that you have enough vRAM for the BGE-M3 embedder (~3-4 GB).
You must set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `GITHUB_TOKEN` in the `.env` file.

### Setup

```bash
uv sync
```

### Pipeline overview

The pipeline consists of four stages: download benchmarks, prepare corpus, build index, run experiment.

#### HotpotQA

```bash
just download-hotpotqa        # Download 500 questions from HuggingFace
just prepare-hotpotqa          # Extract ~500K Wikipedia paragraphs
just index-hotpotqa            # Build ChromaDB index with BGE-M3
just test-hotpot               # Run experiment
```

#### FinanceBench

```bash
just download-financebench       # Download 150 questions from HuggingFace
just download-financebench-pdfs  # Download ~75 SEC filing PDFs from GitHub
just prepare-financebench        # Extract text from all PDF pages via PyMuPDF
just index-financebench          # Build ChromaDB index with BGE-M3
just test-financebench           # Run experiment
```

Or run the full FinanceBench pipeline in one command:

```bash
just pipeline-financebench
```

### Configuration

Experiment configuration files are in `src/retcapslib/`:

- `cfg_test_hotpot.yaml` — HotpotQA experiment config
- `cfg_test_financebench.yaml` — FinanceBench experiment config

To run with a custom config:

```bash
uv run run-experiment --config path/to/config.yaml
```
