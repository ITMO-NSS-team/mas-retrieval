"""Forward function generation — produces Python code for team workflow."""

import json

from langchain_core.prompts import PromptTemplate

from marlib.adapters.swarm_agentic.logger import log
from marlib.adapters.swarm_agentic.prompt.base import FUNCTION_DESCRIPTION

BASE = """You are an expert python programmer.
You are tasked with writing a function to organize available roles to solve a specific task.
{function_description}

You are provided with following available roles. Each role can solve a subtask of the complex task:

<available roles>
{roles}
</available roles>

You are also given the workflow of these roles:

<workflow>
{workflow}
</workflow>

Your job is to design the function that represents how the roles will work together to solve the task.
Use these guidelines when generating the function:
- ALWAYS use **role_response = team.call(role_name: str, inputs: List, output: str)** to call a role. This will give inputs and required output instruction to the role and return the role's response.
    * role_name: The name of the role to call in this step. You can only call roles in the current team. MUST NOT call an unexisting role from available roles.
    * inputs: List of the output produced by one or more roles in previous steps.
    * output: What output expected from the role in this step. Must be enclosed in double quotation marks ("output").
- Use the provided workflow instruction as a guide for designing the function's structure.
- Create a well-organized function that represents how the roles will work together to solve the task efficiently.
- MUST not make any assumptions in the code.
- Ensure that every variable declared in the function is utilized, with no unused or redundant variables.
- Ensure the created function complete and correct to avoid runtime failures.

# Examples:
Here is an examples to help you design the function:

<examples>
{examples}
</examples>
"""

schema = {
    "title": "forward_function",
    "description": "Forward function of agent system to represent the workflow.",
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": '''Design the function in Python code. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets. MUST response in format "def forward(team):\n{Your code here}\nreturn answer".''',
        },
    },
    "required": ["code"],
}


def build_forward(llm, logger, roles, workflow):
    """Generate the forward function via LLM with structured output."""
    prompt = PromptTemplate(
        input_variables=["function_description", "roles", "workflow", "examples"],
        template=BASE,
    )
    chain = prompt | llm.with_structured_output(schema)
    input_vars = {
        "function_description": FUNCTION_DESCRIPTION,
        "roles": roles,
        "workflow": json.dumps(workflow, indent=4),
        "examples": EXAMPLES,
    }
    res = chain.invoke(input_vars)
    log(logger, "Write Forward", prompt.format(**input_vars), res["code"])
    return res["code"]


EXAMPLES = """
Available Roles:
{"Name": "Document Retriever", "Responsibility": "Search the knowledge base for relevant passages", "Policy": "Executes tool automatically."}
{"Name": "Document Reranker", "Responsibility": "Re-rank retrieved passages using cross-encoder", "Policy": "Executes tool automatically."}
{"Name": "Calculator", "Responsibility": "Evaluate mathematical expressions", "Policy": "Executes tool automatically."}
{"Name": "Evidence Analyst", "Responsibility": "Analyze retrieved passages to extract key facts relevant to the question", "Policy": "1. Read all retrieved passages carefully. 2. Identify facts directly relevant to the question. 3. Note any numerical data needed for calculations."}
{"Name": "Answer Synthesizer", "Responsibility": "Synthesize evidence into a concise final answer", "Policy": "1. Review the analysis. 2. Formulate a clear, direct answer. 3. Ensure the answer is supported by evidence."}

Workflow:
[
  {"Step": 1, "Role": "Document Retriever", "Input": "", "Output": "retrieved passages"},
  {"Step": 2, "Role": "Document Reranker", "Input": "retrieved passages", "Output": "reranked passages"},
  {"Step": 3, "Role": "Evidence Analyst", "Input": "reranked passages", "Output": "key facts and analysis"},
  {"Step": 4, "Role": "Answer Synthesizer", "Input": "reranked passages, key facts and analysis", "Output": "final answer"}
]

Answer:
'''def forward(team):
    # Step 1: Document Retriever searches the knowledge base for relevant passages.
    retrieved = team.call('Document Retriever', [], "retrieved passages")

    # Step 2: Document Reranker re-ranks retrieved passages using cross-encoder.
    reranked = team.call('Document Reranker', [retrieved], "reranked passages")

    # Step 3: Evidence Analyst extracts key facts from reranked passages.
    analysis = team.call('Evidence Analyst', [reranked], "key facts and analysis")

    # Step 4: Answer Synthesizer formulates the final answer.
    answer = team.call('Answer Synthesizer', [reranked, analysis], "final answer")

    return answer
'''
"""
