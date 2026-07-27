"""MetaMAS-U: the MetaMAS meta-agent prompts with the size constraint removed.

E1 asks what MetaMAS generates when nothing biases it toward small trees
(T58C's central request; sDic Q1). The bias is entirely prompt-level -- there is
no cap anywhere in AutoMAS' code -- and it is spread over *both* generation
stages, so both system prompts are ablated.

The ablated prompts are derived from the originals by deleting exact blocks
rather than by retyping them. That keeps every other byte identical by
construction, and the deletion is asserted: if AutoMAS' prompts change, this
raises instead of silently ablating nothing (which would make MetaMAS-U a
duplicate of MetaMAS and the experiment a null result for the wrong reason).

Deliberately NOT removed:
  * the few-shot examples (pool: 1- and 2-agent; graph: 1/3/4-agent). They also
    carry the JSON schema, so dropping them would confound the size ablation
    with a format change. They remain a residual size signal -- if |V| does not
    move, they are the prime suspect and the next arm to run.
  * the topology rules (one root, reachability, single terminal, DAG). Those are
    the *structural* half of the constraint, which the paper claims separately;
    E1 ablates size only.
  * "Assign MCP tools only when actually needed" -- bounds tools per agent, not
    the agent count.
"""

from __future__ import annotations

from string import Template

# --- Pool stage (prompt_registry.py:63-70, :105) ------------------------------

_POOL_DESIGN_PRINCIPLES = """\
DESIGN PRINCIPLES:
- START SIMPLE: Create the minimum number of agents needed
- Prefer 1-2 agents for straightforward tasks
- Add more agents only when:
  * Task requires clearly different specialized tools
  * Independent subtasks can be processed in parallel
  * Different expertise domains are needed
- Avoid over-engineering: one versatile agent > multiple similar agents

"""

_POOL_REDUNDANCY_RULE = "- Avoid redundant agents with overlapping capabilities\n"

# --- Graph stage (prompt_registry.py:123-130, :136) ---------------------------

_GRAPH_DESIGN_PRINCIPLES = """\
DESIGN PRINCIPLES:
- SIMPLICITY FIRST: Use the minimum number of agents necessary
- Prefer 1-2 agents for simple tasks over complex multi-step pipelines
- Only add intermediate agents if they provide clear value:
  * Different specialized tools or capabilities needed
  * Parallel processing of independent subtasks
  * Critical data transformation between incompatible formats
- When in doubt, choose the simpler workflow

"""

# A permission rather than a preference, but it is a second, independent size
# reduction: it lets the graph stage execute fewer agents than the pool holds,
# so |V| can shrink after the pool stage has already been ablated. Listed
# separately so the writeup can report it apart from the preference language.
_GRAPH_SUBSET_RULE = "- Select only agents necessary for the task (subset allowed)\n"

POOL_REMOVALS = (_POOL_DESIGN_PRINCIPLES, _POOL_REDUNDANCY_RULE)
GRAPH_REMOVALS = (_GRAPH_DESIGN_PRINCIPLES, _GRAPH_SUBSET_RULE)


def ablate(original: Template, removals: tuple[str, ...], label: str) -> Template:
    """Return *original* with each block in *removals* deleted exactly once.

    Raises if a block is absent, so a prompt change upstream surfaces as a loud
    failure instead of a silently un-ablated arm.
    """
    text = original.template
    for block in removals:
        if block not in text:
            raise RuntimeError(
                f"MetaMAS-U: expected block not found in {label} prompt; AutoMAS' "
                f"prompt_registry.py has changed. Missing block:\n{block!r}"
            )
        text = text.replace(block, "", 1)
    return Template(text)
