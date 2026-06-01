"""Initial-archive reasoning blocks for MAS-Zero (RAG-adapted).

These are the building blocks the meta-agent decomposes a question over: COT,
COT_SC, Reflexion, LLM_debate. Each is a dict {thought, name, code} whose code
defines forward(self, taskInfo). They serve two roles:
  1. the initial archive evaluated directly on the whole question, and
  2. reference implementations the meta-agent must reproduce when wiring
     sub-MAS (the propose prompt forbids calling them by name).

Retrieval (self.retrieve / self.rerank) is embedded into each block so the
sub-MAS can ground its reasoning in the corpus. `Info` and `LLMAgentBase` are
injected into the exec() globals by the adapter — reference them directly, do
not import them. Stdlib imports (e.g. `from collections import Counter`) are ok.
"""

COT = {
    "thought": (
        "Chain-of-thought with retrieval: retrieve relevant documents for the "
        "(sub-)question, then reason step by step over the retrieved context."
    ),
    "name": "COT",
    "code": """def forward(self, taskInfo):
    question = taskInfo.content
    context = self.retrieve(question, top_k=20)
    context_reranked = self.rerank(question, top_k=10)
    context_info = Info('retrieved_context', 'retriever', context_reranked, None, None, None, -1)

    cot_instruction = self.cot_instruction + " Use the retrieved documents as context to support your answer."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)
    thinking, answer = cot_agent([taskInfo, context_info], cot_instruction)

    final_answer = self.make_final_answer(thinking, answer)
    return final_answer
""",
}

COT_SC = {
    "thought": (
        "Self-consistency with retrieval: retrieve once, run several CoT agents "
        "at higher temperature over the same context, and majority-vote answers."
    ),
    "name": "COT_SC",
    "code": """def forward(self, taskInfo):
    from collections import Counter
    question = taskInfo.content
    context = self.retrieve(question, top_k=20)
    context_reranked = self.rerank(question, top_k=10)
    context_info = Info('retrieved_context', 'retriever', context_reranked, None, None, None, -1)

    cot_instruction = self.cot_instruction + " Use the retrieved documents as context to support your answer."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5, usage_callback=self._usage_callback) for _ in range(N)]

    def majority_voting(answers):
        filtered = [a for a in answers if a.strip()]
        if not filtered:
            return ""
        return Counter(filtered).most_common(1)[0][0]

    thinking_mapping = {}
    answer_mapping = {}
    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo, context_info], cot_instruction)
        possible_answers.append(answer.content)
        thinking_mapping[answer.content] = thinking
        answer_mapping[answer.content] = answer

    best = majority_voting(possible_answers)
    if not best or best not in thinking_mapping:
        thinking = Info('thinking', 'fallback', '', None, None, None, -1)
        answer = Info('answer', 'fallback', '', None, None, None, -1)
    else:
        thinking = thinking_mapping[best]
        answer = answer_mapping[best]

    final_answer = self.make_final_answer(thinking, answer)
    return final_answer
""",
}

Reflexion = {
    "thought": (
        "Reflexion with retrieval: produce an initial answer over retrieved "
        "context, get a critic's feedback, re-retrieve with a refined query, "
        "and revise until the critic is satisfied or max_round is reached."
    ),
    "name": "Reflexion",
    "code": """def forward(self, taskInfo):
    question = taskInfo.content
    context = self.retrieve(question, top_k=20)
    context_reranked = self.rerank(question, top_k=10)
    context_info = Info('retrieved_context', 'retriever', context_reranked, None, None, None, -1)

    cot_initial_instruction = self.cot_instruction + " Use the retrieved documents as context to support your answer."
    cot_reflect_instruction = "Given previous attempts, feedback, and retrieved documents, carefully reconsider your answer and improve it based on the feedback."

    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)
    critic_instruction = "Review the answer above given the retrieved context. Criticize where it might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."

    N_max = self.max_round
    cot_inputs = [taskInfo, context_info]
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, context_info, thinking, answer], critic_instruction, i)
        if correct.content.strip() == 'True':
            break
        refined_query = question + " " + feedback.content[:200]
        new_context_reranked = self.rerank(refined_query, top_k=10) if self.retrieve(refined_query, top_k=20) else context_reranked
        context_info = Info('retrieved_context', 'retriever', new_context_reranked, None, None, None, -1)
        cot_inputs = [taskInfo, context_info, thinking, answer, feedback]
        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)

    final_answer = self.make_final_answer(thinking, answer)
    return final_answer
""",
}

LLM_debate = {
    "thought": (
        "Multi-agent debate with retrieval: retrieve shared evidence, have "
        "role-specialised agents debate over several rounds, then a final "
        "decision agent synthesises the answer."
    ),
    "name": "LLM_debate",
    "code": """def forward(self, taskInfo):
    question = taskInfo.content
    context = self.retrieve(question, top_k=20)
    context_reranked = self.rerank(question, top_k=10)
    context_info = Info('retrieved_context', 'retriever', context_reranked, None, None, None, -1)

    debate_initial_instruction = self.cot_instruction + " Use the retrieved documents as evidence for your reasoning."
    debate_instruction = "Given solutions from other agents and the retrieved context, consider their opinions and provide an updated answer with your reasoning."

    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5, usage_callback=self._usage_callback) for role in self.debate_role]
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)
    final_decision_instruction = "Given all the above thinking, answers, and retrieved context, reason carefully and provide a final answer."

    max_round = self.max_round
    all_thinking = [[] for _ in range(max_round)]
    all_answer = [[] for _ in range(max_round)]
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, answer = debate_agents[i]([taskInfo, context_info], debate_initial_instruction)
            else:
                input_infos = [taskInfo, context_info] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, answer = debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)

    thinking, answer = final_decision_agent(
        [taskInfo, context_info] + all_thinking[max_round-1] + all_answer[max_round-1],
        final_decision_instruction,
    )
    final_answer = self.make_final_answer(thinking, answer)
    return final_answer
""",
}

# Initial archive, keyed by the names referenced in the propose prompt.
INIT_BLOCKS = {
    "COT": COT,
    "COT_SC": COT_SC,
    "Reflexion": Reflexion,
    "LLM_debate": LLM_debate,
}


def get_init_archive(block_names: list[str] | None = None) -> list[dict]:
    """Return deep copies of the requested initial-archive blocks."""
    import copy

    names = block_names or list(INIT_BLOCKS)
    return [copy.deepcopy(INIT_BLOCKS[n]) for n in names if n in INIT_BLOCKS]
