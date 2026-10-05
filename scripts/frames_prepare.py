"""Offline snapshot registration and explicit pilot/final ID selection."""
from __future__ import annotations

import argparse
from pathlib import Path

from marlib.benchmarks import discover, get_builder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "split", "questions", "corpus"])
    parser.add_argument("--root", type=Path, default=Path("experiments/benchmarks"))
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--source-url")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, default=Path("configs/frames_ids"))
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--pilot-n", type=int, default=5)
    parser.add_argument("--final-n", type=int, default=200)
    args = parser.parse_args()
    spec = discover(args.root)["frames"]
    from importlib import import_module
    recipe = import_module("_marlib_benchmarks.frames.builder")
    if args.action == "freeze":
        if not all([args.mapping, args.source_url, args.revision]):
            parser.error("freeze requires --mapping, --source-url, --revision")
        recipe.freeze_snapshot(spec.source_dir, args.mapping, args.source_url, args.revision)
    elif args.action == "split":
        selection = recipe.split_ids(spec.load_questions(), args.seed, args.pilot_n, args.final_n)
        args.output.mkdir(parents=True, exist_ok=False)
        recipe.write_json(args.output / "selection.json", {**selection,
                          "questions_sha256": recipe.digest(spec.questions_path.read_bytes())})
        for kind in ["pilot", "final"]:
            recipe.write_json(args.output / f"{kind}.json", selection[f"{kind}_ids"])
    elif args.action == "questions":
        get_builder("frames").download(spec)
    else:
        get_builder("frames").build_corpus(spec)


if __name__ == "__main__":
    main()
