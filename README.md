## Zero-Shot Multi-Agent Generation for Specialized RAG Workflows: An Empirical Evaluation

### Setup

```bash
uv sync
```

Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `GITHUB_TOKEN` in `.env`.

### Run

Experiments are launched with `just run` (cross-platform). All parameters are
flags with defaults in `src/marlib/cli.py`; see them with `just run --help`.

```bash
just run --benchmark financebench --sample-n 10
just run --benchmark hotpotqa --systems fedotmas --note "retriever check"
```
