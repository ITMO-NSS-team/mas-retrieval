# automas_u — MetaMAS without the size constraint

Ablation arm for **E1** of the EMNLP industry rebuttal. Reviewer T58C: *"a claim of the paper is
that constrained tree generation is better than unconstrained, it would be great to have an
ablation version of MetaMAS without the constraints."* Also answers sDic Q1 (is the over-
engineering intrinsic or an artifact of the single-pass cold-start variant?) and, as a byproduct,
sDic Q4 (how stable is the generated tree across generation runs?).

`automas_u` is `automas` with the size-preference language removed from both meta-agent prompts.
Everything else — executor, toolset, retrieval substrate, backbone, temperature (0.3), output
handling, token accounting — is the same code path, because the adapter subclasses
`AutoMASAdapter` and overrides one method. The design rationale is in the module docstrings
(`prompts.py` for what was removed and what was kept on purpose, `adapter.py` for why the patch is
restored in a `finally`).

## Running it

Needs `automas` installed from `automas-research` **branch `main`** — not `emnlp26-industry`, which
renames the package to `metamas` and breaks the adapter's imports. The two branches have a
byte-identical `prompt_registry.py`, so nothing content-bearing differs.

```bash
uv pip install -e ../automas-research/

# corpus-level (CL) — the paper's headline mode
just run --benchmark financebench musique --systems automas automas_u \
    --generation-mode one_time --repeats 3

# query-level (QL) — shows the per-question tail
just run --benchmark financebench musique --systems automas automas_u \
    --generation-mode per_task --repeats 3
```

Run both arms in one command: the CLI holds the retriever, question sample and seed fixed across
systems, so the generated workflow stays the only variable.

## Reading the output

Accuracy and cost land in the usual `results/<run_id>/` files. Workflow shape goes to a sidecar,
one file per arm:

```
results/structure/automas_one_time_financebench.jsonl
results/structure/automas_u_one_time_financebench.jsonl
```

One JSON line per question:

| field | meaning |
|---|---|
| `n_agents` | **\|V\|** — agents wired into the executed pipeline. The headline number. |
| `pool_size` | agents the pool stage generated, *before* the graph stage prunes |
| `n_edges` | \|E\| |
| `critical_path` | longest root-to-leaf path, in nodes (latency; parallel branches do not add) |
| `n_executed` | nodes that actually consumed tokens |
| `instance_id` | unique per adapter instance — **this is how you separate the 3 repeats** |
| `agent_names`, `n_roots`, `n_leaves` | for eyeballing what was generated |

Report the **distribution** of `n_agents`, not just the mean — the point of the comparison is the
tail. (ADAS QL on FinanceBench averages 4.25 calls with sd 17.25 and one query at 214.)

`instance_id` changes per `--repeats` iteration because the CLI builds a fresh adapter each time,
so structural variance across generation runs reads straight off the file — no separate experiment
needed for sDic Q4.

In `one_time` mode the shape is identical for every question within a repeat (one workflow reused),
so the interesting variance is *between* `instance_id`s. In `per_task` it varies per question.

## Interpreting the result

Pre-committed before running, so that every outcome gets reported:

| outcome | reading |
|---|---|
| \|V\|↑, accuracy ≈ or ↓, cost ↑ | the constraint does real work — the paper's claim holds |
| \|V\|↑, accuracy ↑ | uncomfortable but honest; fall back to the cost/accuracy frontier and report it |
| \|V\| unchanged | **do not conclude "the constraint lives deeper than the prompt" yet** — see below |

If `n_agents` does not move, rule out two rival explanations first:

1. **The few-shot examples.** They were deliberately kept (they also carry the JSON schema, so
   removing them would confound size with format), but the pool prompt shows only a 1-agent and a
   2-agent example, which is a size anchor by demonstration. Testing this means a second arm with
   the examples neutralised.
2. **Output truncation.** `BaseMetaAgent` caps generation at `DEFAULT_MAX_TOKENS = 4000`; a large
   pool JSON can hit the ceiling. Check `finish_reason` / output length on the generation call.

Also watch `pool_size` against `n_agents`. If the pool grows but the executed `|V|` does not, the
graph stage is doing the pruning — that localises the constraint to a stage rather than to the
prompts as a whole, which is a finding in its own right.

## Caveat for the writeup

Two of the twenty removed lines are not preference language: the pool's
`- Avoid redundant agents with overlapping capabilities` and the graph's
`- Select only agents necessary for the task (subset allowed)`. The latter is a *permission* rather
than a preference, but it is an independent route to a smaller `|V|`. Report both separately from
the "prefer 1-2 agents" language so the ablation is not overstated.
