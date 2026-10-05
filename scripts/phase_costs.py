"""Decompose instrumented run costs without model calls (standard uncached pricing)."""
import argparse
import json
from pathlib import Path


def summarize(run, prices, reuse_counts=(1, 10, 150, 1000)):
    logs = run["question_logs"]
    if not logs:
        raise ValueError("Run has no questions")
    allowed_model = prices["model"].removeprefix("openai/")
    def price(inp, out):
        return (inp * prices["usd_per_million_input"] + out * prices["usd_per_million_output"]) / 1e6
    phases = {}
    for log in logs:
        if not log.get("resource_summary"):
            raise ValueError("Legacy logs lack phase accounting; do not reconstruct construction cost")
        for call in log["llm_calls"]:
            if call.get("measurement") != "api_attempt":
                raise ValueError("Only instrumented API-attempt logs are supported")
            if call["model"].removeprefix("openai/") != allowed_model:
                raise ValueError(f"No supplied price for {call['model']}")
            phase = phases.setdefault(call["phase"], {"input_tokens": 0, "output_tokens": 0,
                "api_attempts": 0, "unknown_usage_attempts": 0})
            phase["input_tokens"] += call["prompt_tokens"]
            phase["output_tokens"] += call["completion_tokens"]
            phase["api_attempts"] += 1
            phase["unknown_usage_attempts"] += not call["usage_known"]
    for phase in phases.values():
        phase["usd"] = None if phase["unknown_usage_attempts"] else price(phase["input_tokens"], phase["output_tokens"])
    costs = [p["usd"] for p in phases.values()]
    total = None if None in costs else sum(costs)
    construction = phases.get("construction", {}).get("usd", 0.0)
    execution = [v["usd"] for k,v in phases.items() if k != "construction"]
    mean_execution = None if None in execution else sum(execution) / len(logs)
    artifacts = {q.get("artifact_paths", {}).get("workflow") or q.get("artifact_paths", {}).get("construction") for q in logs}
    artifacts.discard(None)
    curve = None
    if run["system_name"].endswith("one_time") and len(artifacts) == 1 and execution and construction is not None and mean_execution is not None:
        curve = {str(n): construction / n + mean_execution for n in reuse_counts if n > 0}
    return {"system": run["system_name"], "questions": len(logs), "phases": phases,
            "system_usd": total, "observed_mean_usd": total / len(logs) if total is not None else None,
            "mean_execution_usd": mean_execution,
            "calculated_CL_reuse_usd_per_answer": curve,
            "reuse_assumption": "Same saved workflow and observed mean execution cost, including failed questions",
            "judge_usd_recorded_usage": price(sum(q.get("judge_prompt_tokens", 0) for q in logs),
                                              sum(q.get("judge_completion_tokens", 0) for q in logs)),
            "price_verified_date": prices["price_verified_date"],
            "limitations": ["Standard uncached tariff; gateway fees and cache discounts excluded",
                            "Recorded judge usage may omit failed judge attempts",
                            "Construction reference summaries are not summed"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--prices", type=Path, default=Path("configs/run_plan.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = summarize(json.loads(args.result.read_text()), json.loads(args.prices.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
