"""Full MAS-Zero adapter for RAG benchmark evaluation.

Per-question, zero-supervision multi-agent design: task decomposition into a
sub-MAS, a meta-iteration reflexion loop driven by MAS-Feedback (solvability +
completeness), and list-wise self-verification. See the `adas` system for the
single-step code-generation baseline.
"""

from .adapter import MASZeroAdapter

__all__ = ["MASZeroAdapter"]
