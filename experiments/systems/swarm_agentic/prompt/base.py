"""Task description and function description for retrieval QA."""

TASK_MINI = """Given a question that requires finding and synthesizing information from a document knowledge base, provide a well-reasoned answer supported by evidence from retrieved passages.

The team has access to the following fixed tool roles:
- "Document Retriever": Searches the knowledge base and returns candidate passages.
- "Document Reranker": Re-scores the most recently retrieved passages using a cross-encoder for better relevance ranking.
- "Calculator": Evaluates a mathematical expression (e.g. "revenue / shares", "round(456.78 / 123, 2)").

Workflow guidelines:
1. Always start by calling "Document Retriever" to get relevant passages.
2. Call "Document Reranker" to refine results.
3. Use "Calculator" for any numerical computations.
4. Reasoning roles analyze the evidence and produce the final answer.

The final answer must be concise and directly address the question."""

FUNCTION_DESCRIPTION = """
The function coordinates a team of specialists to answer a question using a document knowledge base.
The function signature must be 'def forward(team):'.
The team includes fixed tool roles ('Document Retriever', 'Document Reranker', 'Calculator') that access external tools — these MUST be called in the workflow.
The function returns the final answer as a string.
"""

TASK_OUTPUT_SCHEMA = None
