"""Forward function generation and execution for SwarmAgentic."""

from marlib.adapters.swarm_agentic.prompt.write_forward import build_forward


def get_forward(llm, logger, roles, workflow):
    """Generate the forward function code string via LLM.

    Args:
        llm: LLM model to use.
        logger: Logger to record the process.
        roles: String representation of team roles.
        workflow: Workflow structure.

    Returns:
        String of Python code defining ``def forward(team): ...``.
    """
    next_solution = build_forward(llm, logger, roles, workflow)
    return next_solution


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
        raise AssertionError(
            f"{len(names)} things in namespace. Please only provide 1"
        )
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    return func
