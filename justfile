# Download benchmarks (financebench download includes its source PDFs)
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

# Run benchmarks. `just run` is the canonical, cross-OS launcher; pass any CLI
# flag through. See all options with:  just run --help
# e.g.  just run --benchmark hotpotqa --systems fedotmas --sample-n 50 --note "..."
#
# --no-sync: do not let uv prune the environment (some packages are installed
# out-of-band). `python -m marlib.cli` avoids needing the console script
# reinstalled and works on Linux/macOS/Windows.
run *ARGS:
    uv run --no-sync python -m marlib.cli {{ARGS}}

test-hotpot:
    just run --benchmark hotpotqa

test-financebench:
    just run --benchmark financebench
