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

The initial MAS-Zero/FRAMES audit is in `reports/mas_zero_frames_audit.json`; the current proposed run
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

The current implementation is named `mas_zero_rag_self_feedback_v4` in metadata.
It runs generated candidates in supervised workers; `isolate_generated_code: false`
uses the separately labelled `mas_zero_rag_self_feedback_v4_in_process` variant.
It retains seeds, per-question meta-iterations, internal feedback and selection.
It is an adaptation of [upstream MAS-Zero](https://github.com/SalesforceAIResearch/MAS-Zero/tree/66b901264eaf809ed03beaf696d34660ae7de71e),
with differences documented in the audit. `one_time` is rejected.

Version 4 fixes two RAG adaptation defects found in the October 5 QL pilot:
the generated-code example now retrieves before reranking (empty rerank results
are truthy strings), and all four initial blocks pass their actual task and agent
outputs to internal feedback. Rejected meta-model proposals retain their cost and
now record the rejection reason and logical call ID. Search settings and resource
limits are unchanged. Historical v3 runs retain their original metadata; the
affected pilot cannot establish the corrected variant's quality or cost. Verify
a small corrected pilot before running the full comparison.

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

Request admission now covers MetaMAS CL/QL, MAS-Zero, `adas_budgeted`,
`single_agent` and `generated_single_agent`, including MetaMAS MCP primitives. Request/tool counts
are reserved before instrumented calls; output capacity is reserved across
threads. Token limits stop on observed usage, so input tokens can overshoot a
threshold. Unknown usage stops further calls when a token threshold is set.
Async agent phases are cancelled at their wall deadline, with per-request
network timeouts as well. Generated Python in `adas_budgeted` and MAS-Zero v3/v4
runs in separate processes whose process groups are killed on timeout or
budget stop. Model/tool calls go through the parent process and its ledger.
These workers receive no API credentials or gold answer, and permit only the
documented imports/helpers. This is not an OS security sandbox. A cancelled
retrieval thread can finish its computation after the deadline, and cancelling
an API request does not establish that the provider stopped billing. Such
request usage remains unknown. MAS-Zero with `isolate_generated_code: false` and historical `adas` retain their
in-process execution semantics.
CLI rejects resource-limit configurations for unsupported adapters.

On a budget stop, MAS-Zero returns the nonempty completed candidate with highest
self-fitness, or abstains, without another model call. Logs retain the stop and
partial cost. Failed empty answers receive zero answer accuracy; missing judge
verdicts remain missing, with explicit metric denominators. Retrieval call counts
now count `retrieve`/`search`; rerank, calculator, and total tools are separate.
Historical logs retain their original semantics and must not be rewritten.

### Prompt-ablation adapters

- `adas_compact_prompt` adds a compactness instruction to the original ADAS
  generator while retaining its execution policy and retry/fallback behavior.
- `generated_single_agent_legacy` generates a reusable instruction for a single
  agent with the original executor defaults, including the framework's
  50-logical-request limit and default tool scheduling.

Both save model/tool events and generated artifacts per run, with construction
usage accounted separately. Configuration: `configs/finance_prompt_ablation.json`.

On a Linux machine with FinanceBench data and retrieval models already prepared,
run a three-question pilot for each adapter from the repository root:

```sh
sh scripts/run_finance_prompt_pilot.sh
```

### Generated-instruction control

`generated_single_agent` implements the CL instruction control. The generator
receives the benchmark description and the same first three unlabelled examples
used by MetaMAS CL. It creates one instruction per independent repeat, with no
answer feedback or instruction selection. The static and generated conditions
share the Pydantic AI executor, retrieval/rerank/calculator tools, sequential tool
execution, temperature, and execution limits. Generation model defaults to the
executor model; `meta_model` can override it and is recorded in metadata.

Both conditions now record each SDK attempt, including retries and failed calls,
instead of a single aggregate usage event. They default to temperature 0.1,
4096 output tokens per request, and 50 API attempts per answer. These settings
are explicit in the new `single_agent_accounted_v1` condition; historical runs
remain unchanged. On budget exhaustion the agent returns an empty failed answer
without an extra finalization call. Async phases now have cancellation deadlines;
only generated-code workers have process termination.

Construction and execution have separate resource sessions. Generated instruction,
generator inputs/output, errors and event journals are saved in the run artifacts.
Construction calls appear only in the first attempted question's `llm_calls`;
later questions reference the same artifact without duplicating its usage.
Sum call events by phase when calculating construction/reuse cost; do not sum
the repeated `construction_reference` summaries. An empty or truncated generated
instruction fails the repeat: subsequent questions do not trigger another
construction attempt. A new run context or benchmark resets the instruction.

The offline integration checks use Pydantic AI 1.56.0 and real OpenAI SDK clients
with mocked HTTP responses. They cover prompt reuse/reset, attempt counts,
tool/request limits, unknown usage, partial failures and absent gold answers in
model payloads. A minimal environment for these tests can be installed with:

```bash
uv pip install --python .venv/bin/python 'pydantic-ai-slim[openai]==1.56.0'
PYTHONPATH=src .venv/bin/python -m pytest tests/test_single_agent_accounting.py
```

The proposed paired technical pilot uses equal execution limits of 10 API
attempts, 20 tool calls, 50,000 observed tokens and 120 seconds. Construction has
its own limits. After retrieval/environment preparation and authorization for
model calls, its command is:

```bash
just run --benchmark financebench --systems single_agent generated_single_agent \
  --sample-n 5 --generation-mode one_time \
  --adapter-config configs/finance_prompt_control.pilot.json \
  --condition-id finance_prompt_control_pilot \
  --note "technical pilot; no accuracy tuning"
```

The complete matched execution pilot now uses `configs/finance_matched.pilot.json`
for MetaMAS CL, Agentic RAG, generated instruction CL, and `adas_budgeted` CL.
All four use the same execution limits; the three generators also have equal,
separate construction limits. No live pilot has run, and the final budget grid
must be selected from technical usage/feasibility before comparative accuracy.

### MetaMAS and the code-generator control

MetaMAS uses its original PoolGenerator, GraphGenerator, schema validation and
DAG execution. Dedicated tracked clients replace private generator/node clients.
All nodes use the selected executor model, even if the generated pool proposes
a different one; the proposal is saved in `workflow.json`. Construction and
pipeline traces are retained, including a partial pool if graph creation fails.
External framework Python source hashes are recorded in effective metadata.

Each MetaMAS node gets its own MCP stdio server exposing only `retrieve`,
`rerank`, and `calculate`. A hook in the parent reserves the common tool budget
before dispatch, collects source IDs and latency, and journals failures. Tools
within one node run sequentially because rerank uses its preceding retrieval;
nodes within a DAG level retain upstream parallel execution. Budget stops are
latched so upstream exception wrapping cannot hide them or resume requests.
CL construction is charged once; QL uses one full-answer session for generation
and execution. The stop policy is an empty answer without another model call.

`adas_budgeted` is a separate CL condition; original `adas` remains available.
It offers the ADAS RAG seed blocks and the same benchmark description/three
unlabelled examples as MetaMAS. It makes one code proposal, with no accuracy
selection, repair cycle, or fallback to a seed. Invalid construction fails the
repeat. These choices differ from historical ADAS and are recorded as
`adas_budgeted_cl_v1`. All node model/temperature settings are enforced by the
parent; JSON format retries count as model attempts. MAS-Zero v3/v4 uses the same
worker supervisor but preserves its generated node temperatures, five JSON
attempts, candidate search and internal verification. The restricted execution
policy is included in both generators' prompts.

### Run preparation on Linux

The current Intel Mac has no wheel for the required `torch>=2.10`; the user will
run experiments on Linux. Local checks cover the real AutoMAS package, real SDKs
with mocked HTTP, actual MCP subprocesses, and worker termination. They do not
establish GPU/retrieval readiness or provider access on the Linux host.

From the project root on Linux:

```bash
uv sync --group dev --group benchmarks --group comparison
uv pip install --python .venv/bin/python --no-deps -e /path/to/automas-research
uv run --no-sync python -m pytest
```

Replace `/path/to/automas-research` with the local framework checkout. Its older
OpenTelemetry dependency cap conflicts with this harness; `--no-deps` retains
the comparison stack tested here and avoids installing unrelated browser/media
features. The `comparison` group pins Pydantic AI 1.56.0 and FastMCP 2.14.5. Keep
using `--no-sync` after this source installation. The framework source hashes
record the actual checkout used; compatibility must be checked again if it differs.

Reuse the established FinanceBench data/index if available. Otherwise prepare
only FinanceBench (this downloads data and embedding weights, with no LLM calls):

```bash
uv run --no-sync download-benchmarks --benchmark financebench
uv run --no-sync prepare-corpus --benchmark financebench
uv run --no-sync build-index --benchmark financebench
uv run --no-sync python scripts/preflight_finance.py --load-retriever \
  --output reports/finance_preflight_linux.json
```

#### Existing FinanceBench evidence mappings

FinanceBench's [`evidence_page_num` is zero-based](https://github.com/patronus-ai/financebench/blob/main/README.md).
Our corpus uses one-based IDs (`_p1` for the first PDF page). Earlier question
files copied the raw page number into `gold_doc_ids`, pointing at the preceding
page and producing invalid `_p0` IDs. Fix all mappings from the saved raw
`evidence`, including IDs that happened to match an existing corpus page:

```sh
uv run --no-sync python scripts/repair_finance_evidence.py
uv run --no-sync python scripts/repair_finance_evidence.py --apply
uv run --no-sync python scripts/preflight_finance.py --load-retriever \
  --output reports/finance_preflight_linux.json
```

The first command previews the changes. The second backs up the exact original
file as `questions.before_evidence_fix.jsonl` and updates only `gold_doc_ids`.
It preserves question order, answers, raw evidence, corpus and index. Repeating
the repair is a no-op; an existing backup is never overwritten. No downloads or
LLM calls are made. Recompute historical `context_recall` using corrected gold
IDs and saved retrieved document IDs before comparing it with new runs. The
repair does not change answer metrics or retroactively edit result files.

Remaining missing evidence pages require a separate corpus check. Missing
`index_provenance.json` is also a separate issue: the repair cannot establish
which embedding model produced an old index and does not fabricate provenance.

#### Verify an existing index without rebuilding it

If the only remaining problem is missing `index_provenance.json`, run:

```sh
uv run --no-sync python scripts/preflight_finance.py --register-existing-index \
  --output reports/finance_preflight_linux.json
```

This compares every stored document ID and text with the current corpus, checks
cosine distance, and compares up to 16 stored embeddings with fresh BGE-M3
embeddings (cosine similarity must be at least 0.999). Sample IDs are selected by
SHA-256 order, independently of evaluation answers. It also runs the retrieval
query with reranking. Model weights must already be cached; no downloads or LLM
calls are made. Do not update the corpus or index concurrently with this check.

Only after all checks pass does it create the missing sidecar. The record is
marked `verified_existing_index`, includes the verification environment and
sample similarities, and leaves the historical model revision unknown. Sampled
compatibility does not establish how every stored vector was produced. Corpus
texts and indexed vectors are not rebuilt or replaced; existing provenance is
never overwritten. Normal preflight runs remain checks without registration.

The preflight checks all 150 question IDs, evidence-page coverage, corpus/index
hash agreement and a retrieval query. The query runs with offline model loading;
missing weights fail instead of triggering a download. `ready` refers to this
retrieval check; provider access is explicitly untested. Missing evidence must
be resolved without silently dropping questions. Save the data and index along
with the reported hashes. Index model repository revisions are still unpinned.

For these comparisons route all adapters through the same OpenRouter endpoint:
set `OPENAI_BASE_URL=https://openrouter.ai/api/v1` and configure `OPENAI_API_KEY`
and `OPENROUTER_API_KEY` locally. Do not put credentials in run configuration files.
The following commands **make paid model calls** and remain proposed pilots:

```bash
just run --benchmark financebench \
  --systems automas single_agent generated_single_agent adas_budgeted \
  --model openai/gpt-4o-mini --sample-n 5 --generation-mode one_time \
  --adapter-config configs/finance_matched.pilot.json \
  --condition-id finance_matched_pilot --note "technical pilot; no accuracy tuning"

just run --benchmark financebench --systems mas_zero automas single_agent \
  --model openai/gpt-4o-mini --sample-n 5 --generation-mode per_task \
  --adapter-config configs/finance_ql.pilot.json \
  --condition-id finance_ql_pilot --note "technical pilot; no accuracy tuning"
```

These are 20 and 15 answers respectively. The second Agentic RAG condition uses
the larger competitor budget and is intentionally a separate pilot condition.
Do not pool it with the smaller-budget baseline. The five FinanceBench pilot
questions overlap the proposed full-150 final set; do not tune on their accuracy.

Decompose a saved system result by phase without any API calls:

```bash
uv run --no-sync python scripts/phase_costs.py results/RUN/SYSTEM.json \
  --output reports/phase_costs.json
```

The script sums actual call events, charges CL construction once, and produces
a calculated reuse curve from the saved workflow and mean execution cost.
Unknown usage makes the affected cost unknown. Legacy aggregate logs are rejected.
The USD figures use the dated standard uncached tariff in `configs/run_plan.json`,
exclude gateway fees/cache discounts, and show recorded judge usage separately.
