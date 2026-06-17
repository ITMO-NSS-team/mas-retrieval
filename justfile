# List discovered benchmarks and systems (only systems with installed deps appear).
available:
    uv run --no-sync python -c "from marlib.benchmarks import discover; from marlib.adapters import discover_adapters; print('benchmarks:', list(discover())); print('systems:', discover_adapters())"

# Prepare benchmarks: download -> corpus -> index; already-done steps skip themselves.
# Pass one or more names, or nothing / "all" for every discovered benchmark.
# E.g. just prepare hotpotqa  |  just prepare hotpotqa financebench  |  just prepare all
prepare *names:
    #!/usr/bin/env sh
    set -eu
    names="{{names}}"
    if [ -z "$names" ] || [ "$names" = "all" ]; then
        names=$(uv run --no-sync python -c "from marlib.benchmarks import discover; print(' '.join(discover()))")
    fi
    for name in $names; do
        echo ">>> preparing $name"
        uv run download-benchmarks --benchmark "$name"
        uv run prepare-corpus --benchmark "$name"
        uv run build-index --benchmark "$name"
    done

# Build LightRAG indexes separately: this uses LLM calls during indexing.
# Extra flags can be passed directly with `uv run build-lightrag-index`.
lightrag-index *names:
    #!/usr/bin/env sh
    set -eu
    names="{{names}}"
    if [ -z "$names" ] || [ "$names" = "all" ]; then
        names=$(uv run --no-sync python -c "from marlib.benchmarks import discover; print(' '.join(discover()))")
    fi
    for name in $names; do
        echo ">>> LightRAG indexing $name"
        uv run --group lightrag build-lightrag-index --benchmark "$name"
    done

# Run experiments (cross-OS); pass any CLI flag. See all with: just run --help
run *ARGS:
    uv run --no-sync python -m marlib.cli {{ARGS}}
