"""Structural metrics for a generated AutoMAS workflow.

The paper's complexity measure is the shape of the generated agent tree: its size
|V|, its edge count |E|, and its critical-path length (``main.tex:77``). None of
that is recoverable from ``QuestionLog.num_llm_calls``: the AutoMAS adapter logs
one aggregated LLM call per question, so that field is the constant 1 whatever
the tree looks like. These helpers read the shape off the built pipeline instead.

Everything here works on a plain sequence of nodes exposing ``id``, ``name``,
``parents`` and ``children`` (i.e. ``automas.pipeline.node.AgentNode``), so the
functions are testable with stubs and never import AutoMAS.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence


class _Node(Protocol):
    """The slice of ``AgentNode`` these metrics need."""

    id: Any
    name: str
    parents: Sequence[Any]
    children: Sequence[Any]


def edge_count(nodes: Iterable[_Node]) -> int:
    """Number of edges |E|, counted once per (parent -> child) pair."""
    return sum(len(node.children) for node in nodes)


def critical_path_length(nodes: Sequence[_Node]) -> int:
    """Longest root-to-leaf path, in nodes.

    Mirrors ``Pipeline._compute_execution_levels``: a node with no parents is at
    level 0 and any other node sits one below its deepest parent, so the number
    of levels is the longest path. Counted in nodes (a single-agent tree is 1),
    matching how the paper reports one model call for a one-node workflow.

    Returns 0 for an empty pipeline. Cycles cannot occur (``GraphGenerator``
    validates the graph as a DAG) but are handled defensively: a node already
    being visited contributes no depth rather than recursing forever.
    """
    if not nodes:
        return 0

    depth: dict[Any, int] = {}
    visiting: set[Any] = set()

    def node_depth(node: _Node) -> int:
        if node.id in depth:
            return depth[node.id]
        if node.id in visiting:  # defensive: a cycle would otherwise recurse
            return 0
        visiting.add(node.id)
        parents = list(node.parents)
        result = 1 if not parents else 1 + max(node_depth(p) for p in parents)
        visiting.discard(node.id)
        depth[node.id] = result
        return result

    return max(node_depth(node) for node in nodes)


def structure_metrics(nodes: Sequence[_Node]) -> dict[str, Any]:
    """Shape of one generated workflow.

    ``n_agents`` is |V| as the paper defines it -- the agents actually wired into
    the executed pipeline. Note this can be smaller than the generated pool:
    ``DEFAULT_GRAPH_INSTRUCT`` (``prompt_registry.py:136``) lets the graph stage
    select a subset of the pool, so callers that care about the generator's raw
    output should record the pool size separately.

    ``n_executed`` counts nodes that actually consumed tokens. The paper asserts
    "the number of model calls is |V|" (``main.tex:77``); a gap between
    ``n_executed`` and ``n_agents`` means some agent never ran, which would make
    that identity an overstatement.
    """
    nodes = list(nodes)
    return {
        "n_agents": len(nodes),
        "n_edges": edge_count(nodes),
        "critical_path": critical_path_length(nodes),
        "n_roots": sum(1 for n in nodes if not list(n.parents)),
        "n_leaves": sum(1 for n in nodes if not list(n.children)),
        "n_executed": sum(
            1
            for n in nodes
            if (getattr(n, "input_tokens", 0) or 0)
            + (getattr(n, "output_tokens", 0) or 0)
            > 0
        ),
        "agent_names": [n.name for n in nodes],
    }
