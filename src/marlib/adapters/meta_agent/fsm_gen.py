"""FSM generation for MetaAgent: agent descriptions + finite state machine."""

from __future__ import annotations

import json
import logging

from marlib.adapters.meta_agent.llm import LLM

logger = logging.getLogger(__name__)

RAG_TOOLS = ["retrieve", "rerank", "calculate"]


def generate_agent_description(
    task: str,
    tools: list[str],
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> tuple[list[dict], LLM]:
    """Generate agent descriptions for the given task and tools."""
    prompt_template = (
        "You are the designer of a multi-agent system. Given a general task "
        "description, you first need to design several agents that can "
        "cooperately solve this type of task.\n"
        "Each agent contain three features:\n"
        "- name: <The name of the agent>\n"
        "- system_prompt: <The system prompt for agent, describe the overall "
        "goal, its name and role, and its responsibility and constrain.>\n"
        "- tools: <The equiped tool name, a list>\n"
        "You are required to define an agent in json format.\n"
        "You answer should obey the following format:\n"
        "Task Analyse: What specific scenarios the task covers? How to design "
        "agents that can adapt to these scenarios\n"
        "System Goal Design:\n"
        "Agents Define:\n"
        "```json\n"
        '[{{"agent_id":"0","name":<fill in agent0\'s name>,'
        '"system_prompt":<fill in agent0\'s system_prompt>,'
        '"tools":[<tool1>,<tool2>,...]}},\n'
        '{{"agent_id":"1","name":<fill in agent1\'s name>,'
        '"system_prompt":<fill in agent1\'s system_prompt>,'
        '"tools":[..]}},\n'
        "...]\n"
        "```\n\n"
        "You can't design too many redundant agents. More Agents means more "
        "cost! Agents need to cooperate efficiently"
    )

    agent_generator = LLM(
        prompt_template, model=model, base_url=base_url, api_key=api_key,
    )
    agents = agent_generator.chat(
        message=(
            f"The General Task is {task} and the tools you can select are "
            f"{tools}. One agent can equipped with various tools and it can "
            "also act without tools. Not every tool is necessary for the task!"
        )
    )

    agent_json = agents.split("```")[-2].replace("json", "")
    agent_dict = json.loads(agent_json)
    return agent_dict, agent_generator


def generate_fsm(
    task: str,
    agent_dict: list[dict],
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> tuple[dict | None, LLM]:
    """Generate a Finite State Machine for the given task and agents."""
    prompt_template = (
        "You are the designer of a multi-agent system. Given a general task "
        "description and a list of agents, you need to generate a Finite "
        "State Machine (FSM) to manage the process of solving the task.\n\n"
        "WARNING: You are good at controlling costs, too many agents and too "
        "complex cooperation structure can lead to excessive costs of "
        "information exchange\n"
        "Each state in the FSM should include:\n"
        "1. state_id: A unique identifier for the state\n"
        "2. agent_id: The ID of the agent associated with this state\n"
        "3. instruction: What the agent should do in this state\n"
        "4. is_initial: Boolean indicating if this is the initial state\n"
        "5. is_final: Boolean indicating if this is a final state\n"
        "6. listener: The agent who will save this state output information "
        "in their memory\n"
        "             Notice : Make sure the listener covers all related "
        "agents. The agents not listed as a listener would not received the "
        "information(which may cause the failure of cooperation)\n"
        "             Hence, some important milestone like a new version of "
        "code/answer should be broadcast all related agent!\n\n"
        "The FSM should also include transition functions between states. "
        "Each transition function should specify:\n"
        "1. from_state: The ID of the state this transition is from\n"
        "2. to_state: The ID of the state this transition goes to\n"
        "3. condition: A description of the condition that triggers this "
        "transition\n\n"
        "Your answer should follow this format:\n"
        "Reasoning: <Your step-by-step reasoning process>\n"
        "Answer:\n"
        "```json\n"
        "{{\n"
        '  "states": [\n'
        "    {{\n"
        '      "state_id": "1",\n'
        '      "agent_id": "0",\n'
        '      "instruction": "Perform task X",\n'
        '      "is_initial": true,\n'
        '      "is_final": false,\n'
        '      "listener":["1","2"]\n'
        "    }},\n"
        "    ...\n"
        "  ],\n"
        '  "transitions": [\n'
        "    {{\n"
        '      "from_state": "1",\n'
        '      "to_state": "2",\n'
        '      "condition": "If task X is completed successfully"\n'
        "    }},\n"
        "    {{\n"
        '      "from_state": "2",\n'
        '      "to_state": "1",\n'
        '      "condition": "If the previous task needs to be re-done."\n'
        "    }},\n"
        "    ...\n"
        "  ]\n"
        "}}\n"
        "```\n\n"
        "Rules:\n"
        "1. Ensure there is exactly one initial state and at least one final "
        "state.\n"
        "2. Every non-final state should have at least one outgoing "
        "transition.\n"
        "3. The FSM should be able to handle loops and complex interactions "
        "between agents.\n"
        "4. Include a transition to a final state that submits the final "
        "answer (use <|submit|> in the instruction).\n"
        "5. Make sure all agent_ids in the states correspond to the provided "
        "agent_dict.\n"
        "6. The transitions should consider as many as possible situations. "
        "Which consisit a roadmap for Multi-Agent System in deployment stage.\n"
        "7. IMPORTANT: Different states should use different agents where "
        "possible. Avoid assigning all states to a single agent — that "
        "defeats the purpose of a multi-agent system."
    )

    fsm_generator = LLM(
        prompt_template, model=model, base_url=base_url, api_key=api_key,
    )
    fsm_response = fsm_generator.chat(
        message=(
            f"The task is: {task}\n"
            f"The agents are: {json.dumps(agent_dict, indent=2)}\n"
            "Now generate the FSM"
        )
    )

    fsm_json = fsm_response.split("```")[1].strip()
    if "<|submit|>" in fsm_json:
        fsm_json = fsm_json.replace(
            "<|submit|>",
            "Use <|submit|> <FILL IN THE FINAL ANSWER> format to submit "
            "the final answer",
        )

    if fsm_json.startswith("json"):
        fsm_json = fsm_json[4:].strip()

    try:
        fsm = json.loads(fsm_json)
    except json.JSONDecodeError:
        logger.warning("Unable to parse FSM JSON from LLM response")
        return None, fsm_generator

    if validate_fsm(fsm, agent_dict):
        return fsm, fsm_generator
    return None, fsm_generator


def validate_fsm(fsm: dict, agent_dict: list[dict]) -> bool:
    """Validate a generated FSM against the agent definitions."""
    initial_states = [s for s in fsm["states"] if s["is_initial"]]
    if len(initial_states) != 1:
        logger.warning("Expected 1 initial state, found %d", len(initial_states))
        return False

    final_states = [s for s in fsm["states"] if s["is_final"]]
    if not final_states:
        logger.warning("No final state found")
        return False

    valid_agent_ids = {a["agent_id"] for a in agent_dict}
    for state in fsm["states"]:
        if state["agent_id"] not in valid_agent_ids:
            logger.warning(
                "Invalid agent_id %s in state %s",
                state["agent_id"], state["state_id"],
            )
            return False

    state_ids = {s["state_id"] for s in fsm["states"]}
    outgoing = {sid: 0 for sid in state_ids}
    for t in fsm["transitions"]:
        if t["from_state"] not in state_ids or t["to_state"] not in state_ids:
            logger.warning("Invalid state ID in transition %s", t)
            return False
        outgoing[t["from_state"]] += 1

    for state in fsm["states"]:
        if not state["is_final"] and outgoing[state["state_id"]] == 0:
            logger.warning(
                "Non-final state %s has no outgoing transitions",
                state["state_id"],
            )
            return False

    # Warn (but don't reject) if all non-final states use the same agent
    non_final = [s for s in fsm["states"] if not s["is_final"]]
    if len(non_final) > 1:
        agents_used = {s["agent_id"] for s in non_final}
        if len(agents_used) == 1:
            logger.warning(
                "FSM degenerate: all %d non-final states use agent %s",
                len(non_final),
                agents_used.pop(),
            )

    return True


def generate_mas(
    task: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[list[dict], dict]:
    """Generate a full MAS (agents + FSM) for the task. Retries once on failure."""
    for attempt in range(2):
        agent_dict, _ = generate_agent_description(
            task=task,
            tools=RAG_TOOLS,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        logger.info("Generated %d agents (attempt %d)", len(agent_dict), attempt + 1)

        fsm, _ = generate_fsm(
            task=task,
            agent_dict=agent_dict,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        if fsm is not None:
            return agent_dict, fsm

        logger.warning("FSM generation failed (attempt %d), retrying...", attempt + 1)

    raise RuntimeError("Failed to generate valid FSM after 2 attempts")
