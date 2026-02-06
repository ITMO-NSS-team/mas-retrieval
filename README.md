## Evaluating Auto-Generated Multi-Agent Systems on Retrieval Tasks

This repo contains code for preparing data and evaluating auto-generated multi-agent systems on retrieval tasks.
Before you need to ensure that you have enought of vRAM.
You must set to `.env` file `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

1. Downloading benchmarks

```bash
download-benchmarks
```

2. Preparing HotpotQA and MuSiQue benchmarks

```bash
prepare-corpus
```

3. Indexing HotpotQA and MuSiQue via BAAI/bge-m3 embedder:

```bash
build-index --dataset hotpotqa --batch-size 32
```

4. Setup configuration at the `config.yaml` and set the path (default path is `./src/retcapslib/config_test.yaml`):

```bash
run-experiment
```
