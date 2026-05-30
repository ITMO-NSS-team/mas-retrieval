from __future__ import annotations

import math

import pydantic_monty

# pi/e are injected as inputs since monty's stdlib subset excludes math.
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expr: str) -> float:
    # monty is a sandboxed interpreter: no imports, attribute escapes, or I/O.
    result = pydantic_monty.Monty(expr.strip(), inputs=list(_CONSTS)).run(inputs=_CONSTS)
    return float(result)


def do_calculate(expression: str) -> str:
    try:
        result = safe_eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"
