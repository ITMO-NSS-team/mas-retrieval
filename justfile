# MAS Retrieval — experiment pipeline commands

# Download benchmarks
download: download-hotpotqa download-financebench

download-hotpotqa:
    uv run download-benchmarks --benchmark hotpotqa

download-financebench:
    uv run download-benchmarks --benchmark financebench

# Prepare corpora
prepare: prepare-hotpotqa prepare-financebench

prepare-hotpotqa:
    uv run prepare-corpus --dataset hotpotqa

prepare-financebench:
    uv run prepare-corpus --dataset financebench

# Build indexes
index-all: index-hotpotqa index-financebench

index-hotpotqa:
    uv run build-index --dataset hotpotqa

index-financebench:
    uv run build-index --dataset financebench

# Run tests
test-hotpot:
    uv run run-experiment --config src/retcapslib/cfg_test_hotpot.yaml

test-financebench:
    uv run run-experiment --config src/retcapslib/cfg_test_financebench.yaml

# Full pipelines
pipeline-financebench: download-financebench prepare-financebench index-financebench test-financebench
