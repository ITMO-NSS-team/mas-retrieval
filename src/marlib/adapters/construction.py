"""Unlabelled construction input shared by comparison conditions."""


def task_description(description: str | None, questions: list[str]) -> str:
    parts = [description or "Answer questions accurately using retrieval-augmented generation."]
    if questions:
        examples = "\n".join(f"- {q}" for q in questions[:3])
        parts.append(f"\nExample questions from the benchmark:\n{examples}")
    return "\n".join(parts)
