"""Forward function generation and execution for SwarmAgentic."""

import re
from collections.abc import Iterable

from .logger import log
from .prompt.write_forward import build_forward
from .role import normalize_name

# Matches team.call('Role Name') / team.call("Role Name")
_CALL_RE = re.compile(r"""team\.call\(\s*['"]([^'"]+)['"]""")


def extract_called_roles(code: str) -> list[str]:
    """Extract all role names referenced via ``team.call(...)`` in the code."""
    return _CALL_RE.findall(code)


def get_forward(llm, logger, roles, workflow, valid_names, max_retries=3):
    """Generate the forward function code string via LLM, validated against roles.

    Regenerates the code if it references roles that do not exist in the team
    (compared with normalized matching), feeding the offending names back to the
    LLM so it can correct them.

    Args:
        llm: LLM model to use.
        logger: Logger to record the process.
        roles: String representation of team roles.
        workflow: Workflow structure.
        valid_names: Iterable of valid role names callable in the team.
        max_retries: Maximum number of generation attempts.

    Returns:
        String of Python code defining ``def forward(team): ...``.
    """
    valid_names = list(valid_names)
    valid_norm = {normalize_name(n) for n in valid_names}

    feedback = ""
    next_solution = ""
    for _ in range(max_retries):
        next_solution = build_forward(llm, logger, roles, workflow, feedback)
        unknown = _unknown_roles(next_solution, valid_norm)
        if not unknown:
            return next_solution
        feedback = (
            "Your previous code called these non-existent roles: "
            f"{sorted(unknown)}. You MUST only call roles from this exact list, "
            f"using the names verbatim: {valid_names}."
        )
        log(logger, "Forward Validation", feedback, mark="-")

    log(
        logger,
        "Forward Validation",
        f"Giving up after {max_retries} attempts; "
        f"unresolved roles: {sorted(_unknown_roles(next_solution, valid_norm))}",
        mark="-",
    )
    return next_solution


def _unknown_roles(code: str, valid_norm: Iterable[str]) -> set[str]:
    """Return called role names not present in the set of valid normalized names."""
    valid_norm = set(valid_norm)
    return {
        name
        for name in extract_called_roles(code)
        if normalize_name(name) not in valid_norm
    }


def set_forward(next_solution):
    """Compile a Python code string into a callable function.

    Args:
        next_solution: Python code defining exactly one callable.

    Returns:
        The callable defined by the code.

    Raises:
        AssertionError: If not exactly one callable is defined.
    """
    namespace = {}
    exec(next_solution, globals(), namespace)  # noqa: S102
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    return func
