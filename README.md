## Evaluating Auto-Generated Multi-Agent Systems on QA & RAG Tasks

`marlib` (in `src/`) is the **harness**: retriever, tracing, evaluation, CLI, and
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
just prepare financebench
```

### Run

Parameters are flags with defaults in `src/marlib/cli.py` (`just run --help`).

```bash
just run --benchmark financebench --sample-n 10
just run --benchmark hotpotqa --systems fedotmas --note "retriever check"
```
