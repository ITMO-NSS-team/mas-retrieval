from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from marlib.retriever.core import Document, Retriever


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No results found."
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.title} (score: {doc.score:.3f})\n{doc.text}")
    return "\n\n".join(parts)


def do_retrieve(
    retriever: Retriever,
    query: str,
    top_k: int = 20,
) -> tuple[list[Document], str]:
    docs = retriever.retrieve(query, top_k=top_k)
    return docs, format_docs(docs)


def do_rerank(
    retriever: Retriever,
    query: str,
    docs: list[Document],
    top_k: int = 10,
) -> tuple[list[Document], str]:
    reranked = retriever.rerank(query, docs, top_k=top_k)
    return reranked, format_docs(reranked)


# Calculator: AST-based safe eval, no builtins or attribute access.
_SAFE_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
}
_SAFE_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _SAFE_CONSTS:
        return _SAFE_CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _SAFE_FUNCS
    ):
        fn_name = node.func.id
        args = [_eval_node(a) for a in node.args]
        if fn_name == "round" and len(args) == 2:
            args[1] = int(args[1])
        return float(_SAFE_FUNCS[fn_name](*args))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def safe_eval(expr: str) -> float:
    tree = ast.parse(expr.strip(), mode="eval")
    return _eval_node(tree.body)


def do_calculate(expression: str) -> str:
    try:
        result = safe_eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"
