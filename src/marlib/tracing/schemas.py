from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Record of a single tool invocation during question execution."""

    tool_name: str = Field(description="Name of the tool (retrieve, rerank, calculate)")
    query: str = Field(description="Query string passed to the tool")
    top_k: int = Field(description="Number of results requested")
    results: list[str] = Field(description="List of returned doc_ids")
    latency_ms: float = Field(description="Tool execution latency in milliseconds")


class LLMCall(BaseModel):
    """Record of a single LLM API call during question execution."""

    model: str = Field(description="Model identifier (e.g., gpt-4o-mini)")
    prompt_tokens: int = Field(description="Number of prompt tokens")
    completion_tokens: int = Field(description="Number of completion tokens")
    latency_ms: float = Field(description="API call latency in milliseconds")
    function_calls: int = Field(
        default=0, description="Number of function/tool calls in response"
    )


class QuestionLog(BaseModel):
    """Complete execution log for a single question."""

    question_id: str = Field(description="Unique identifier for the question")
    question: str = Field(description="The question text")
    gold_answer: str = Field(description="Ground truth answer")
    predicted_answer: str = Field(description="System's predicted answer")

    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="All tool calls made"
    )
    llm_calls: list[LLMCall] = Field(
        default_factory=list, description="All LLM API calls made"
    )

    total_prompt_tokens: int = Field(
        default=0, description="Total prompt tokens across all LLM calls"
    )
    total_completion_tokens: int = Field(
        default=0, description="Total completion tokens across all LLM calls"
    )
    total_tokens: int = Field(
        default=0, description="Total tokens (prompt + completion)"
    )
    total_latency_ms: float = Field(
        default=0.0, description="Total execution time in milliseconds"
    )

    num_retrieval_calls: int = Field(
        default=0, description="Number of retrieval tool calls"
    )
    num_llm_calls: int = Field(default=0, description="Number of LLM API calls")

    error: str | None = Field(default=None, description="Error message if execution failed")

    # Evaluation scores, keyed by metric name (the set is declared per benchmark
    # in its manifest; a metric that does not apply to a question is omitted).
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Per-question metric scores by name"
    )


class SystemResults(BaseModel):
    """Aggregated results for a single system on a benchmark."""

    system_name: str = Field(description="Name of the MAS auto-generator system")
    benchmark: str = Field(description="Benchmark name (hotpotqa, musique)")
    model: str = Field(description="LLM model used")

    question_logs: list[QuestionLog] = Field(
        default_factory=list, description="Per-question execution logs"
    )

    # Aggregate evaluation scores, averaged over questions, keyed by metric name.
    avg_metrics: dict[str, float] = Field(
        default_factory=dict, description="Average score per metric by name"
    )

    avg_tokens_per_question: float = Field(
        default=0.0, description="Average total tokens per question"
    )
    avg_prompt_tokens_per_question: float = Field(
        default=0.0, description="Average prompt (input) tokens per question"
    )
    avg_completion_tokens_per_question: float = Field(
        default=0.0, description="Average completion (output) tokens per question"
    )
    avg_retrieval_calls: float = Field(
        default=0.0, description="Average retrieval calls per question"
    )
    avg_llm_calls: float = Field(
        default=0.0, description="Average LLM calls per question"
    )
    avg_latency_ms: float = Field(
        default=0.0, description="Average latency per question in ms"
    )

    total_questions: int = Field(default=0, description="Total questions evaluated")
    failed_questions: int = Field(
        default=0, description="Questions that failed with errors"
    )
