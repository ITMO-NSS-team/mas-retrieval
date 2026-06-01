"""ADAS adapter for RAG benchmark evaluation.

Single-step meta-agent code generation (ADAS / Meta Agent Search style): the
meta-model designs ONE agentic architecture from building blocks, which is then
executed per question. This is NOT MAS-Zero — it omits MAS-Zero's meta-iteration
feedback loop, task decomposition, and self-verification. See the `mas_zero`
system for the full MAS-Zero algorithm.
"""

from .adapter import ADASAdapter

__all__ = ["ADASAdapter"]
