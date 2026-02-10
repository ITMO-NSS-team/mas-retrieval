"""RAG-adapted reasoning blocks for MAS-Zero.

Each block is a dict with {thought, name, code} where code defines a
forward(self, taskInfo) method. The blocks use self.retrieve(query, top_k)
and self.rerank(query, top_k) provided by AgentSystem for document retrieval.

These are adapted from MAS-Zero's original blocks (COT, Reflexion, Debate,
COT-SC) with retrieval operations embedded into the reasoning flow.
"""

RAG_COT = {
    "thought": (
        "Chain-of-thought reasoning enhanced with retrieval: first retrieve "
        "relevant documents for the question, then reason step by step over "
        "the retrieved context to produce an answer."
    ),
    "name": "RAG-Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Extract question text from taskInfo
    question = taskInfo.content

    # Retrieve relevant documents
    context = self.retrieve(question, top_k=20)
    context_reranked = self.rerank(question, top_k=10)

    # Build context info for the agent
    from retcapslib.adapters.mas_zero.core import Info, LLMAgentBase
    context_info = Info('retrieved_context', 'retriever', context_reranked, None, None, None, -1)

    cot_instruction = self.cot_instruction + " Use the retrieved documents as context to support your answer."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'RAG-CoT Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)

    thinking, answer = cot_agent([taskInfo, context_info], cot_instruction)
    final_answer = self.make_final_answer(thinking, answer)
    return final_answer
""",
}

RAG_REFLEXION = {
    "thought": (
        "Reflexion with retrieval: retrieve documents, generate an initial "
        "answer, get feedback from a critic, and optionally re-retrieve with "
        "a refined query before producing an improved answer."
    ),
    "name": "RAG-Reflexion",
    "code": """def forward(self, taskInfo):
    from retcapslib.adapters.mas_zero.core import Info, LLMAgentBase

    question = taskInfo.content

    # Initial retrieval
    context = self.retrieve(question, top_k=20)
    context_reranked = self.rerank(question, top_k=10)
    context_info = Info('retrieved_context', 'retriever', context_reranked, None, None, None, -1)

    cot_initial_instruction = self.cot_instruction + " Use the retrieved documents as context to support your answer."
    cot_reflect_instruction = "Given previous attempts, feedback, and retrieved documents, carefully reconsider your answer. Try to improve based on the feedback."

    cot_agent = LLMAgentBase(['thinking', 'answer'], 'RAG-CoT Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)
    critic_instruction = "Review the answer above given the retrieved context. Criticize where it might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."

    N_max = self.max_round
    cot_inputs = [taskInfo, context_info]
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, context_info, thinking, answer], critic_instruction, i)
        if correct.content.strip() == 'True':
            break

        # Re-retrieve with refined query based on feedback
        refined_query = question + " " + feedback.content[:200]
        new_context = self.retrieve(refined_query, top_k=20)
        new_context_reranked = self.rerank(refined_query, top_k=10)
        context_info = Info('retrieved_context', 'retriever', new_context_reranked, None, None, None, -1)

        cot_inputs = [taskInfo, context_info, thinking, answer, feedback]
        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)

    final_answer = self.make_final_answer(thinking, answer)
    return final_answer
""",
}

RAG_DEBATE = {
    "thought": (
        "LLM debate enhanced with retrieval: retrieve documents, then have "
        "multiple agents with different roles debate using the retrieved "
        "context, before a final decision agent synthesizes the best answer."
    ),
    "name": "RAG-LLM-Debate",
    "code": """def forward(self, taskInfo):
    from retcapslib.adapters.mas_zero.core import Info, LLMAgentBase

    question = taskInfo.content

    # Retrieve documents for debate context
    context = self.retrieve(question, top_k=20)
    context_reranked = self.rerank(question, top_k=10)
    context_info = Info('retrieved_context', 'retriever', context_reranked, None, None, None, -1)

    debate_initial_instruction = self.cot_instruction + " Use the retrieved documents as evidence for your reasoning."
    debate_instruction = "Given solutions from other agents and the retrieved context, consider their opinions. Provide an updated answer with your thinking process."

    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5, usage_callback=self._usage_callback) for role in self.debate_role]

    final_decision_instruction = "Given all the above thinking, answers, and retrieved context, reason carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)

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

RAG_COT_SC = {
    "thought": (
        "Self-consistency with retrieval: retrieve documents, then run "
        "multiple chain-of-thought agents over the same context with higher "
        "temperature, and select the most common answer via majority voting."
    ),
    "name": "RAG-Self-Consistency-COT",
    "code": """def forward(self, taskInfo):
    from collections import Counter
    from retcapslib.adapters.mas_zero.core import Info, LLMAgentBase

    question = taskInfo.content

    # Retrieve documents
    context = self.retrieve(question, top_k=20)
    context_reranked = self.rerank(question, top_k=10)
    context_info = Info('retrieved_context', 'retriever', context_reranked, None, None, None, -1)

    cot_instruction = self.cot_instruction + " Use the retrieved documents as context to support your answer."
    N = self.max_sc

    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'RAG-CoT Agent', model=self.node_model, temperature=0.5, usage_callback=self._usage_callback) for _ in range(N)]

    def majority_voting(answers):
        return Counter(answers).most_common(1)[0][0]

    thinking_mapping = {}
    answer_mapping = {}
    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo, context_info], cot_instruction)
        possible_answers.append(answer.content)
        thinking_mapping[answer.content] = thinking
        answer_mapping[answer.content] = answer

    best = majority_voting(possible_answers)
    thinking = thinking_mapping[best]
    answer = answer_mapping[best]

    final_answer = self.make_final_answer(thinking, answer)
    return final_answer
""",
}

# All available RAG blocks
RAG_BLOCKS = [RAG_COT, RAG_REFLEXION, RAG_DEBATE, RAG_COT_SC]
