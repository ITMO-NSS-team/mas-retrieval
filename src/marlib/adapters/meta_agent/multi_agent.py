"""FSM-based multi-agent orchestrator adapted from MetaAgent."""

from __future__ import annotations

import re
from typing import Any, Callable

from marlib.adapters.meta_agent.llm import LLM


class MultiAgentSystem:
    """Runs an FSM-driven team of LLM agents with tool-call dispatch."""

    def __init__(
        self,
        agents_json: list[dict[str, Any]],
        states_json: dict[str, Any],
        tool_executor: Callable[..., str],
        model: str,
        base_url: str | None,
        api_key: str | None,
        tracker: Any | None = None,
    ) -> None:
        self.agents = {a["agent_id"]: a for a in agents_json}
        self.states = {s["state_id"]: s for s in states_json["states"]}
        self.transitions = states_json["transitions"]
        self.listeners = {
            s["state_id"]: s.get("listener", []) for s in states_json["states"]
        }
        self._tool_executor = tool_executor
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._tracker = tracker
        self.llms: dict[str, LLM] = {}
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        for agent_id, agent in self.agents.items():
            updated = self._update_system_prompt(agent)
            self.llms[agent_id] = LLM(
                system_prompt=updated["system_prompt"],
                model=self._model,
                base_url=self._base_url,
                api_key=self._api_key,
                tracker=self._tracker,
            )

    def _update_system_prompt(self, agent: dict[str, Any]) -> dict[str, Any]:
        tools_desc = ""
        if agent.get("tools"):
            tools_desc = (
                "\nYou can use these tools via <tool_call> tags:\n"
                '- retrieve: <tool_call>retrieve(query="your search query", '
                "top_k=20)</tool_call>\n"
                '- rerank: <tool_call>rerank(query="your query", '
                "top_k=10)</tool_call>\n"
                '- calculate: <tool_call>calculate(expression="1234 * 0.15")'
                "</tool_call>\n"
            )

        transition_conditions = ""
        for t in self.transitions:
            if (
                t["from_state"] in self.states
                and self.states[t["from_state"]]["agent_id"] == agent["agent_id"]
            ):
                transition_conditions += (
                    f"- If {t['condition']}, output "
                    f"`<STATE_TRANS>: {t['to_state']}`.\n"
                )

        transition_conditions += (
            "- If no conditions are met, output `<STATE_TRANS>: None`.\n"
            " DO NOT WRITE THIS IN THE CODE SNIPPET!"
        )

        agent["system_prompt"] += tools_desc + "\n" + transition_conditions
        return agent

    # ── Tool-call parsing ──────────────────────────────────────

    _TOOL_CALL_RE = re.compile(
        r"<tool_call>\s*(\w+)\(([^)]*)\)\s*</tool_call>", re.DOTALL,
    )
    _KWARG_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|(\d+))')

    def _extract_tool_calls(
        self, output: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        calls: list[tuple[str, dict[str, Any]]] = []
        for m in self._TOOL_CALL_RE.finditer(output):
            name = m.group(1)
            raw_args = m.group(2)
            kwargs: dict[str, Any] = {}
            for km in self._KWARG_RE.finditer(raw_args):
                key = km.group(1)
                val: Any = km.group(2) if km.group(2) is not None else int(km.group(3))
                kwargs[key] = val
            calls.append((name, kwargs))
        return calls

    # ── Info extraction ────────────────────────────────────────

    @staticmethod
    def _extract_info(output: str) -> str:
        start_tag = "<INFO>"
        end_tag = "</INFO>"
        si = output.find(start_tag)
        ei = output.find(end_tag)
        if si != -1 and ei != -1:
            return output[si + len(start_tag) : ei]
        return output

    # ── FSM transition ─────────────────────────────────────────

    def _get_next_state(self, current_state: str, output: str) -> str | None:
        for t in self.transitions:
            if (
                t["from_state"] == current_state
                and f"<STATE_TRANS>: {t['to_state']}" in output
            ):
                return t["to_state"]
        return None

    # ── Agent execution ────────────────────────────────────────

    def _run_agent(
        self,
        state_id: str,
        input_data: str | None = None,
        max_transitions: int = 10,
        transition_count: int = 0,
        ini_flag: int = 0,
    ) -> str:
        state = self.states[state_id]
        agent = self.agents[state["agent_id"]]
        llm = self.llms[agent["agent_id"]]

        if state["is_initial"] and ini_flag == 0:
            instruction = state["instruction"] + "\nThe user input is:\n" + (input_data or "")
            for all_llm in self.llms.values():
                all_llm.add_message("The user input is:\n" + (input_data or ""))
        elif ini_flag == 0:
            instruction = state["instruction"]
        else:
            instruction = input_data or ""

        instruction += (
            "\nAdd <STATE_TRANS>: <fill in id> after complete the task "
            "and make sure the tool is executed successfully"
        )

        conversation_count = 0
        empty_retrieval_count = 0
        output = ""
        while conversation_count < 4:
            output = llm.chat(instruction)

            if not output:
                output = " "

            if "<|submit|>" in output or state["is_final"]:
                try:
                    return output.split("<|submit|>")[1].strip()
                except IndexError:
                    return output

            next_state_id = self._get_next_state(state_id, output)

            # Execute tool calls if present
            tool_calls = self._extract_tool_calls(output)
            if tool_calls:
                results_parts: list[str] = []
                for name, kwargs in tool_calls:
                    try:
                        result = self._tool_executor(name, **kwargs)
                    except Exception as e:
                        result = f"Error calling {name}: {e}"
                    results_parts.append(f"[{name}] {result}")

                tool_output = "\n".join(results_parts)
                all_empty = all(
                    "No results found" in r for r in results_parts
                )
                hint = ""
                if all_empty:
                    empty_retrieval_count += 1
                    if empty_retrieval_count >= 3:
                        hint = (
                            "\nCRITICAL: Retrieval has returned no results "
                            f"{empty_retrieval_count} times. STOP retrying. "
                            "Answer with your best knowledge and transition "
                            "to the next state using <STATE_TRANS>: <id>."
                        )
                    else:
                        hint = (
                            "\nIMPORTANT: Previous retrieval returned no results. "
                            "Try a DIFFERENT, simpler query. Break multi-hop "
                            "questions into simpler sub-queries."
                        )
                else:
                    empty_retrieval_count = 0
                instruction = (
                    f"Tool results:\n{tool_output}{hint}\n\n"
                    "After completing the current step, please use "
                    "<STATE_TRANS>: <fill in states id> and pass necessary "
                    "information to the next agent."
                )
                conversation_count += 1
                continue

            if next_state_id:
                transition_count += 1
                if transition_count >= max_transitions:
                    return output

                info = self._extract_info(output)
                for listener_id in self.listeners.get(state_id, []):
                    if listener_id in self.llms:
                        self.llms[listener_id].add_message(
                            "Message from " + agent["name"] + "\n" + info
                        )
                return self._run_agent(
                    next_state_id, info, max_transitions, transition_count,
                )

            instruction = (
                "After completed current step task, please use "
                "<STATE_TRANS>: <fill in states id>"
            )
            conversation_count += 1

        # Loop exhausted — attempt final state transition before returning
        next_state_id = self._get_next_state(state_id, output)
        if next_state_id:
            transition_count += 1
            if transition_count < max_transitions:
                info = self._extract_info(output)
                for listener_id in self.listeners.get(state_id, []):
                    if listener_id in self.llms:
                        self.llms[listener_id].add_message(
                            "Message from " + agent["name"] + "\n" + info
                        )
                return self._run_agent(
                    next_state_id, info, max_transitions, transition_count,
                )

        return self._extract_info(output)

    # ── Public entry point ─────────────────────────────────────

    def start(self, user_input: str, max_transitions: int = 10) -> tuple[str, int]:
        """Run the FSM on user_input. Returns (result, token_cost)."""
        initial_costs = {
            aid: llm.get_token_cost() for aid, llm in self.llms.items()
        }

        initial_state = next(
            s for s in self.states.values() if s["is_initial"]
        )
        result = self._run_agent(initial_state["state_id"], user_input, max_transitions)

        total_cost = sum(
            llm.get_token_cost() - initial_costs[aid]
            for aid, llm in self.llms.items()
        )
        return result, total_cost
