"""Recompute the proposed run estimate without model calls or network access."""
import argparse
import json
from pathlib import Path


def estimate(plan):
    def cost(usage):
        return (usage["input"] * plan["usd_per_million_input"] +
                usage["output"] * plan["usd_per_million_output"]) / 1e6
    results = {}
    for stage in ("pilot", "recommended_final"):
        rows, total_low, total_high, hours_low, hours_high = [], 0, 0, 0, 0
        for grid in plan[stage]:
            if not grid.get("enabled", True):
                continue
            for system in grid["systems"]:
                n = grid["questions"] * grid["repeats"]
                measured = system == "mas_zero"
                usages = plan["historical_mas_zero_pilots"] if measured else [plan["assumed_per_answer_usage"][system]]
                construction = plan["assumed_cl_construction"] if system.endswith("_cl") else {"input": 0, "output": 0, "seconds": 0}
                judge = plan["assumed_external_judge"]
                costs = [n * cost(u) + grid["repeats"] * cost(construction) for u in usages]
                hours = [(n * (u["seconds"] + judge["seconds"]) + grid["repeats"] * construction["seconds"]) / 3600 for u in usages]
                judge_cost = n * cost(judge)
                row = {"benchmark": grid["benchmark"], "condition_id": grid.get("condition_id"), "system": system, "answers": n,
                       "basis": "historical_pilot_extrapolation" if measured else "explicit_planning_assumption",
                       "system_usd": [min(costs), max(costs)], "judge_usd": judge_cost,
                       "serial_hours_including_judge": [min(hours), max(hours)]}
                rows.append(row)
                total_low += min(costs) + judge_cost
                total_high += max(costs) + judge_cost
                hours_low += min(hours)
                hours_high += max(hours)
        results[stage] = {"rows": rows, "answers": sum(r["answers"] for r in rows),
                          "total_usd": [total_low, total_high],
                          "planning_reserve_usd": total_high * (1 + plan["contingency_fraction"]),
                          "serial_hours": [hours_low, hours_high]}
    mas_prices = [cost(p) for p in plan["historical_mas_zero_pilots"]]
    results["optional_mas_zero_frames_200x3"] = {
        "system_usd": [600 * min(mas_prices), 600 * max(mas_prices)],
        "warning": "Assumes FinanceBench usage transfers to FRAMES; excluded from recommended package"}
    results["price_source"] = plan["price_source"]
    results["price_verified_date"] = plan["price_verified_date"]
    results["deferred_grids"] = {stage: [g for g in plan[stage] if not g.get("enabled", True)]
                                for stage in ("pilot", "recommended_final")}
    results["assumptions"] = plan["assumptions"]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/run_plan.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(estimate(json.loads(args.plan.read_text())), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")
