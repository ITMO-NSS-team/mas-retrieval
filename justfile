# Prepare a benchmark: download -> corpus -> index; already-done steps skip themselves. E.g. just prepare hotpotqa
prepare name:
    uv run download-benchmarks --benchmark {{name}}
    uv run prepare-corpus --benchmark {{name}}
    uv run build-index --benchmark {{name}}

# Run experiments (cross-OS); pass any CLI flag. See all with: just run --help
run *ARGS:
    uv run --no-sync python -m marlib.cli {{ARGS}}
