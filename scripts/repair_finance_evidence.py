"""Repair FinanceBench evidence IDs offline, without changing corpus or index."""
import argparse
import json
from pathlib import Path

from marlib.benchmarks import discover, get_builder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/benchmarks/financebench"))
    parser.add_argument("--apply", action="store_true", help="Write corrected questions after creating an exclusive backup")
    args = parser.parse_args()
    discover()
    result = get_builder("financebench").repair_evidence_ids(args.root, apply=args.apply)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
