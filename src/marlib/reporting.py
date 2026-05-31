"""Cross-repeat summary table.

Renders one ``system × benchmark`` table whose cells stack four quantities —
``llm_accuracy``, total input tokens, total output tokens, and recomputed cost —
each as ``mean ± std`` over the repeats of a run. Cost is derived from the token
totals via the ``genai-prices`` catalogue.
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.table import Table

from marlib.log import logger
from marlib.tracing.schemas import SystemResults

_ACCURACY_METRIC = "llm_accuracy"


def run_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Total USD cost of one run's token usage via ``genai-prices``.

    ``model`` may be provider-prefixed (``openai/gpt-4o-mini``); the prefix is
    passed as the provider hint. Returns ``None`` if the model is unknown to the
    price catalogue (so the caller can render ``—`` rather than crash).
    """
    from genai_prices import Usage, calc_price

    provider, _, name = model.partition("/")
    if not name:  # no '/' -> the whole string is the model, no provider hint
        provider, name = "", model
    try:
        calc = calc_price(
            Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            model_ref=name,
            provider_id=provider or None,
        )
        return float(calc.total_price)
    except Exception as e:
        logger.warning(f"cost: no price for model '{model}': {e}")
        return None


def _run_token_totals(r: SystemResults) -> tuple[int, int]:
    """(input, output) token totals summed over the run's question logs."""
    in_tok = sum(log.total_prompt_tokens for log in r.question_logs)
    out_tok = sum(log.total_completion_tokens for log in r.question_logs)
    return in_tok, out_tok


def _judge_token_totals(r: SystemResults) -> tuple[int, int]:
    """(input, output) LLM-judge token totals over the run's question logs."""
    in_tok = sum(log.judge_prompt_tokens for log in r.question_logs)
    out_tok = sum(log.judge_completion_tokens for log in r.question_logs)
    return in_tok, out_tok


def _stat(values: list[float | None]) -> tuple[float, float | None, int]:
    """(mean, sample std or None when <2 points, n) over the non-None values."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return (float("nan"), None, 0)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n >= 2 else None
    return (mean, std, n)


def _htok(v: float) -> str:
    return f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"


def _hcost(v: float) -> str:
    return f"{v:.2f}" if v >= 1 else f"{v:.4f}"


def _pm(mean: float, std: float | None, fmt) -> str:
    """Format ``mean ± std`` (just ``mean`` when std is None, i.e. a single run)."""
    return fmt(mean) if std is None else f"{fmt(mean)}±{fmt(std)}"


def _cell(results: list[SystemResults], model: str, judge_model: str) -> str:
    """A cell (acc / in / out / system $ / judge $) for one (system, benchmark)."""
    if not results:
        return "—"
    accs: list[float | None] = [r.avg_metrics.get(_ACCURACY_METRIC) for r in results]
    ins: list[float | None] = []
    outs: list[float | None] = []
    costs: list[float | None] = []
    jcosts: list[float | None] = []
    judge_tok_total = 0
    for r in results:
        i, o = _run_token_totals(r)
        ins.append(i)
        outs.append(o)
        costs.append(run_cost(model, i, o))
        ji, jo = _judge_token_totals(r)
        judge_tok_total += ji + jo
        jcosts.append(run_cost(judge_model, ji, jo))

    am, asd, an = _stat(accs)
    im, isd, _ = _stat(ins)
    om, osd, _ = _stat(outs)
    cm, csd, cn = _stat(costs)
    jm, jsd, _ = _stat(jcosts)

    acc = "acc —" if an == 0 else f"acc {_pm(am, asd, lambda x: f'{x:.3f}')}"
    return "\n".join(
        [
            acc,
            f"in  {_pm(im, isd, _htok)}",
            f"out {_pm(om, osd, _htok)}",
            "$   —" if cn == 0 else f"$   {_pm(cm, csd, _hcost)}",
            # Judge cost is "—" when llm_accuracy was not scored (no judge tokens).
            "j$  —" if judge_tok_total == 0 else f"j$  {_pm(jm, jsd, _hcost)}",
        ]
    )


def render_summary(
    collected: dict[tuple[str, str], list[SystemResults]],
    benchmarks: list[str],
    repeats: int,
    model: str,
    judge_model: str,
    console: Console | None = None,
) -> None:
    """Print the ``system × benchmark`` summary table to the console.

    Args:
        collected: per-(system, benchmark) list of ``SystemResults``, one entry
            per successful repeat.
        benchmarks: benchmark names, used as the column order.
        repeats: number of repeats requested (for the title and a completeness note).
        model: system-under-test model id, for the title and system cost lookup.
        judge_model: LLM-judge model id, for the judge cost lookup.
        console: optional Rich console (defaults to a fresh stdout console).
    """
    console = console or Console()
    # Row order = first-seen system; column order = benchmarks as requested.
    systems: list[str] = []
    for system_name, _bench in collected:
        if system_name not in systems:
            systems.append(system_name)
    if not systems:
        logger.warning("summary: nothing to report (no successful runs).")
        return

    table = Table(
        title=(
            f"Summary — {repeats} repeat(s), model={model}, judge={judge_model}\n"
            "cell: acc=llm_accuracy / in,out=total tokens / $=system cost / "
            "j$=judge cost, mean±std over repeats"
        ),
        show_lines=True,
    )
    table.add_column("system", style="bold")
    for b in benchmarks:
        table.add_column(b, justify="left")

    incomplete: list[str] = []
    for s in systems:
        row = [s]
        for b in benchmarks:
            results = collected.get((s, b), [])
            if 0 < len(results) < repeats:
                incomplete.append(f"{s}/{b} ({len(results)}/{repeats})")
            row.append(_cell(results, model, judge_model))
        table.add_row(*row)

    console.print(table)
    if incomplete:
        console.print(
            f"[yellow]note:[/yellow] fewer than {repeats} successful repeats for: "
            + ", ".join(incomplete)
        )
