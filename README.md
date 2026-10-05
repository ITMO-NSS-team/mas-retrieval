## Evaluating Auto-Generated Multi-Agent Systems on QA & RAG Tasks

Tiny library `marlib` is the **harness**: retriever, tracing, evaluation, CLI, and the adapter/benchmark contracts + discovery. The **content** it measures lives outside the package and is discovered by path:

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

Content deps are opt-in groups in `pyproject.toml` — one per system plus a `benchmarks` group for the builders. `fedotmas`/`automas` install from local source. Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `GITHUB_TOKEN` in `.env`.

### Prepare a benchmark

Downloads questions/sources, builds the corpus, and indexes it. Already-done steps are skipped.

```bash
just prepare hotpotqa
just prepare hotpotqa financebench  # several, space-separated
just prepare                        # all discovered benchmarks (same as: just prepare all)
```

List what's available with `just available` (discovered benchmarks and systems).

**BioASQ needs a manual data drop.** Register at [participants-area.bioasq.org](https://participants-area.bioasq.org/datasets/), grab a **Task B** training file (e.g. `training14b.json`), and drop it in:

```bash
mkdir -p experiments/benchmarks/bioasq/source
cp /path/to/training14b.json experiments/benchmarks/bioasq/source/
just prepare bioasq
```

`prepare` then samples 250 factoid + 250 list questions (stratified, seeded) and fetches the gold PubMed abstracts as the corpus, so `doc_id == PubMed ID`. Until the file is present, `just prepare` (all benchmarks) just skips BioASQ with a note.

### Run

Parameters are flags with defaults in `src/marlib/cli.py` (`just run --help`).

```bash
just run --benchmark financebench --sample-n 10
just run --benchmark hotpotqa --systems naive_rag fedotmas --note "retriever check"
just run --benchmark financebench --systems fedotmas --model openai/gpt-4o     # pick the LLM (default: openai/gpt-4o-mini)
just run --benchmark hotpotqa musique financebench bioasq --systems naive_rag  # one system, every benchmark
just run --benchmark hotpotqa musique --systems naive_rag single_agent --repeats 5
```

**Generation mode.** Systems that auto-generate a MAS can do it once for the whole benchmark or fresh for every question, pick with `--generation-mode`:

```bash
just run --benchmark financebench --systems fedotmas --generation-mode one_time  # generate once, reuse across the benchmark
just run --benchmark financebench --systems fedotmas --generation-mode per_task  # regenerate the MAS for each question
```

**Judge.** The `llm_accuracy` metric is scored by a fixed LLM judge (`openai/gpt-4o-mini` by default, independent of `--model`); override with `JUDGE_MODEL` in `.env`.

### AAMAS extension: audit and offline checks

The current audit is in `reports/mas_zero_frames_audit.json`; the proposed run
matrix is in `configs/run_plan.json`. Recompute its estimate without API calls:

```bash
python scripts/estimate_runs.py --output reports/run_estimate.json
PYTHONPATH=src .venv/bin/python -m pytest
```

The offline environment used for this audit contains only test/parser/client
dependencies. Full retrieval still requires the normal `uv sync`, model weights,
and a prepared index. `logly` is constrained below 0.2 because 0.2 changed the
configuration API used here.

### FRAMES preparation

The builder pins Google FRAMES revision
`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef` and verifies the TSV SHA256.
It reads **all** link fields, including `wikipedia_link_11+` and `wiki_links`.
The local protocol uses the full frozen MLCommons Wikipedia corpus with the
shared BGE retrieval stack. It is a separate protocol from MLPerf scoring.

Obtain the frozen `docs.tar.gz` and `url_mapping.json` from
[MLCommons storage](https://inference.mlcommons-storage.org/frames-benchmark-dataset/doc_html/docs.tar.gz).
The checked archive has 173,138,122 bytes and SHA256
`f5e6d7c14f93cdd0af49f9f72e2311419af7c9b0634755fb2310058a84e36e80`.
Extract its HTML files to `experiments/benchmarks/frames/source/doc_html/`, and put
the mapping at `experiments/benchmarks/frames/source/url_mapping.json`.
The [MLCommons preparation script](https://github.com/mlcommons/inference/blob/3fbc329939999c13d0a7b5e67fb2092287e06047/e2e-rag/scripts/download_dataset_and_models.sh)
documents these assets; its full model-download step is unnecessary for our stack.

```bash
PYTHONPATH=src uv run --no-sync python scripts/frames_prepare.py questions
PYTHONPATH=src uv run --no-sync python scripts/frames_prepare.py freeze \
  --mapping experiments/benchmarks/frames/source/url_mapping.json \
  --source-url https://inference.mlcommons-storage.org/frames-benchmark-dataset/doc_html/docs.tar.gz \
  --revision sha256:f5e6d7c14f93cdd0af49f9f72e2311419af7c9b0634755fb2310058a84e36e80
PYTHONPATH=src uv run --no-sync python scripts/frames_prepare.py corpus
```

`freeze` hashes each local HTML file. `corpus` verifies those hashes, preserves
table rows/list items, resolves canonical URLs, and writes `coverage.json`,
`sources.json`, and `corpus_provenance.json`. Missing/ambiguous evidence stops the
build; questions are retained. Review the coverage report before indexing.
No live Wikipedia fallback is used. Complex table spans are not expanded into a
relational schema; tabular retrieval quality still needs a pilot.

**Coverage audit, 2026-10-02:** all 2515 HTML files parsed successfully, yielding
2486 distinct canonical articles and 72,002 chunks. The frozen corpus lacks 30
source URLs needed by 13 questions, including four frozen final-sample IDs.
`reports/frames_coverage.json` lists every gap. Corpus publication and indexing
are blocked until this is resolved; the builder deliberately fails on the current
snapshot. Twenty-eight missing links use the mobile Wikipedia domain, which the
examined upstream download regex does not match; two other mapped URLs have no
HTML file. Keep the sampled IDs fixed while resolving this data issue.

The checked-in `configs/frames_ids/` freezes seed 20261002, five pilot IDs and
200 disjoint final IDs. To propose a different sample, use a **new** directory:

```bash
PYTHONPATH=src uv run --no-sync python scripts/frames_prepare.py split \
  --output configs/frames_ids_alternative --seed 20261002 --pilot-n 5 --final-n 300
```

After successful coverage, build the shared index with `uv run build-index
--benchmark frames`. Keep the full corpus when running an ID subset. Index
provenance records the corpus hash, embedder, collection and package versions;
embedding model repository revisions are not yet pinned.

### MAS-Zero variant and resource accounting

The evaluated implementation is named `mas_zero_rag_self_feedback_v2` in metadata.
It retains seeds, per-question meta-iterations, internal feedback and selection.
It is an adaptation of [upstream MAS-Zero](https://github.com/SalesforceAIResearch/MAS-Zero/tree/66b901264eaf809ed03beaf696d34660ae7de71e),
with differences documented in the audit. `one_time` is rejected.

For a separately authorized technical pilot after environment/index preparation:

```bash
just run --benchmark financebench --systems mas_zero --sample-n 5 \
  --generation-mode per_task --adapter-config configs/mas_zero.pilot.json \
  --condition-id mas_zero_technical_pilot --note "technical pilot; no accuracy tuning"
```

Use `--question-ids configs/frames_ids/pilot.json` for a prepared FRAMES pilot;
it is mutually exclusive with `--sample-n`. An adapter-config JSON is keyed by
registered system name. CLI metadata records requested/effective settings,
ordered IDs, source/data hashes and installed package versions.

MAS-Zero records actual attempts (SDK retries disabled), logical request IDs,
usage availability, measured request latency, and phases `construction`,
`answer_execution`, `internal_verification`. The external judge stays separate.
Events and candidates are saved during execution under a unique directory for
each question attempt inside the run. Per-question JSONL checkpoints are written
after evaluation. A hard process interruption can leave unmatched start events;
those requests have unknown usage and require reconciliation.

The current budgets are **cooperative and MAS-Zero-only**: request/tool counts
are reserved before instrumented calls; output capacity is reserved across
threads. Token limits stop on observed usage, so input tokens can overshoot a
threshold. Unknown usage stops further calls when a token threshold is set.
Wall time is checked at call boundaries with a per-request network timeout;
it does not kill arbitrary generated Python. Generated code can also bypass
helpers by importing clients directly. Process isolation and a common transport
for external framework/MCP calls remain prerequisites for matched-budget claims.
CLI rejects resource-limit configurations for unsupported adapters.

On a budget stop, MAS-Zero returns the nonempty completed candidate with highest
self-fitness, or abstains, without another model call. Logs retain the stop and
partial cost. Failed empty answers receive zero answer accuracy; missing judge
verdicts remain missing, with explicit metric denominators. Retrieval call counts
now count `retrieve`/`search`; rerank, calculator, and total tools are separate.
Historical logs retain their original semantics and must not be rewritten.
