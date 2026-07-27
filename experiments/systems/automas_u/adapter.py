"""MetaMAS-U -- MetaMAS with the generation size constraint removed (E1).

The ablation changes exactly one thing: the two meta-agent system prompts. It
subclasses the MetaMAS adapter rather than copying it, so the executor, toolset,
retrieval substrate, output handling, token accounting and structure logging are
not merely equivalent but the *same code path*. Backbone, temperature (0.3, from
AutoMAS' own generator defaults) and the run protocol come from the CLI
unchanged.

Run both arms together::

    just run --benchmark financebench --systems automas automas_u \\
        --generation-mode one_time --repeats 3

|V|, |E| and the critical path land in ``results/structure/`` -- one file per
arm, since the sidecar is named after ``self.name``.
"""

from __future__ import annotations

from typing import Any

from marlib.adapters.base import register

from ..automas.adapter import AutoMASAdapter
from .prompts import GRAPH_REMOVALS, POOL_REMOVALS, ablate


@register("automas_u")
class AutoMASUnconstrainedAdapter(AutoMASAdapter):
    """MetaMAS with the size-preference language stripped from both prompts."""

    @property
    def name(self) -> str:
        return f"automas_u_{self._generation_mode}"

    async def _ensure_structure(self, question: str) -> tuple[Any, Any]:
        """Generate with the ablated prompts, then put the originals back.

        The prompts are module-level globals in AutoMAS, so patching them is
        process-wide. Restoring in a ``finally`` keeps the ablation from leaking
        into a baseline ``automas`` run in the same process -- ``cli.py`` runs
        the requested systems sequentially in one process, so without this
        ``--systems automas_u automas`` would silently ablate the baseline too.

        Patching per call (rather than once at import) also means the block
        assertions in ``ablate()`` always run against pristine prompts.
        """
        import automas.meta_agents.graph_gen as graph_gen
        import automas.meta_agents.pool_gen as pool_gen

        original_pool = pool_gen.DEFAULT_POOL_INSTRUCT
        original_graph = graph_gen.DEFAULT_GRAPH_INSTRUCT

        pool_gen.DEFAULT_POOL_INSTRUCT = ablate(original_pool, POOL_REMOVALS, "pool")
        graph_gen.DEFAULT_GRAPH_INSTRUCT = ablate(
            original_graph, GRAPH_REMOVALS, "graph"
        )
        try:
            return await super()._ensure_structure(question)
        finally:
            pool_gen.DEFAULT_POOL_INSTRUCT = original_pool
            graph_gen.DEFAULT_GRAPH_INSTRUCT = original_graph
