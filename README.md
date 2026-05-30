## Evaluating Auto-Generated Multi-Agent Systems on QA & RAG Tasks

Tiny library `marlib` is the **harness**: retriever, tracing, evaluation, CLI, and
the adapter/benchmark contracts + discovery. The **content** it measures lives
outside the package and is discovered by path:

```
experiments/
  systems/<name>/       # a system under test: __init__.py + adapter.py
  benchmarks/<name>/    # a benchmark: manifest.toml + builder.py (+ generated data)
```

Add a system or benchmark by dropping in a folder — no library edits.

### Setup

```bash
uv sync                                            # harness only
uv sync --group benchmarks --group swarm_agentic   # + content you actually run
```

Content deps are opt-in groups in `pyproject.toml` — one per system plus a
`benchmarks` group for the builders. `fedotmas`/`automas` install from local
source. Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `GITHUB_TOKEN` in `.env`.

### Prepare a benchmark

Downloads questions/sources, builds the corpus, and indexes it. Already-done
steps are skipped.

```bash
just prepare hotpotqa
just prepare hotpotqa financebench  # several, space-separated
just prepare                        # all discovered benchmarks (same as: just prepare all)
```

List what's available with `just available` (discovered benchmarks and systems).

**BioASQ needs a manual data drop.** Its Task B dataset is gated behind a free
account, so there's no automatic download. Register at
[participants-area.bioasq.org](https://participants-area.bioasq.org/datasets/),
grab a **Task B** training file (e.g. `training14b.json` — it carries the
`question / gold PubMed documents / answer` triples; the other tasks are indexing
or NER, not QA), and drop it in:

```bash
mkdir -p experiments/benchmarks/bioasq/source
cp /path/to/training14b.json experiments/benchmarks/bioasq/source/
export NCBI_EMAIL="you@example.com"   # optional: polite/faster PubMed efetch (set NCBI_API_KEY for 10 req/s)
just prepare bioasq
```

`prepare` then samples 250 factoid + 250 list questions (stratified, seeded) and
fetches the gold PubMed abstracts as the corpus, so `doc_id == PubMed ID`. Until
the file is present, `just prepare` (all benchmarks) just skips BioASQ with a note.

### Run

Parameters are flags with defaults in `src/marlib/cli.py` (`just run --help`).

```bash
just run --benchmark financebench --sample-n 10
just run --benchmark hotpotqa --systems naive_rag fedotmas --note "retriever check"
just run --benchmark financebench --systems fedotmas --model openai/gpt-4o  # pick the LLM (default: openai/gpt-4o-mini)
```

Pass several systems space-separated after `--systems` to run them in one process.

**Generation mode.** Systems that auto-generate a MAS can do it once for the whole
benchmark or fresh for every question — pick with `--generation-mode`:

```bash
just run --benchmark financebench --systems fedotmas --generation-mode one_time  # generate once, reuse across the benchmark
just run --benchmark financebench --systems fedotmas --generation-mode per_task  # regenerate the MAS for each question
```

Omit the flag to use each adapter's default. The mode is recorded in the run's
provenance and appended to the system name in results (e.g. `fedotmas_one_time`),
so the two modes never overwrite each other.
