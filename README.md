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

### Run

Parameters are flags with defaults in `src/marlib/cli.py` (`just run --help`).

```bash
just run --benchmark financebench --sample-n 10
just run --benchmark hotpotqa --systems naive_rag fedotmas --note "retriever check"
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
