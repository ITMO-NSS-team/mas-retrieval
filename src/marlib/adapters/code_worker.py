"""Restricted generated-code worker. JSON RPC on stdin/stdout; no API credentials.

This is process separation and an execution policy, not an OS security sandbox.
The supervisor owns model/tool calls, budgets, artifacts and the deadline.
"""
from __future__ import annotations

import ast
import builtins
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import resource
import sys
import types

ALLOWED_IMPORTS = {"math", "random", "statistics", "collections"}
ALLOWED_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
    "len", "list", "max", "min", "range", "reversed", "round", "set", "sorted",
    "str", "sum", "tuple", "zip", "Exception", "ValueError", "RuntimeError", "print",
)


def validate_code(code):
    tree = ast.parse(code)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in tree.body):
        raise ValueError("Generated code must define forward(self, taskInfo)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_") and node.attr != "_usage_callback":
            raise ValueError("Generated code cannot access private attributes")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Generated code cannot access dunder names")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef)):
            raise ValueError("Unsupported generated-code statement")
        if isinstance(node, ast.Import):
            if any(alias.name not in ALLOWED_IMPORTS for alias in node.names):
                raise ValueError("Only math/random/statistics/collections imports are permitted")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS or any(a.name.startswith("_") or a.name == "*" for a in node.names):
                raise ValueError("Unsupported generated import")
    compile(tree, "<generated>", "exec")


def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in ALLOWED_IMPORTS or any(n.startswith("_") for n in fromlist):
        raise ImportError("Import outside generated-code policy")
    return builtins.__import__(name, globals, locals, fromlist, level)


def main():
    wire_out, wire_in = sys.stdout, sys.stdin
    def send(value):
        wire_out.write(json.dumps(value) + "\n")
        wire_out.flush()
    def rpc(kind, **payload):
        send({"kind": kind, **payload})
        line = wire_in.readline()
        if not line:
            raise RuntimeError("Supervisor closed the worker channel")
        return json.loads(line)["result"]
    try:
        request = json.loads(wire_in.readline())
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
        validate_code(request["code"])
        # Trusted local runtime only. Imports by generated code are separately limited.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        spec = importlib.util.spec_from_file_location("generated_runtime", request["core_path"])
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        def model_request(messages, model, output_fields, temperature, usage_callback=None):
            return rpc("llm", messages=messages, output_fields=output_fields, temperature=temperature)
        core._get_json_response = model_request
        system = core.AgentSystem()
        for key, value in request["system_config"].items():
            setattr(system, key, value)
        system._usage_callback = None
        system._retrieve_fn = lambda query, top_k=20: rpc("retrieve", query=query, top_k=top_k)
        system._rerank_fn = lambda query, top_k=10: rpc("rerank", query=query, top_k=top_k)
        system._calc_fn = lambda expression: rpc("calculate", expression=expression)
        safe_builtins = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}
        safe_builtins["__import__"] = limited_import
        namespace = {"__builtins__": safe_builtins, "LLMAgentBase": core.LLMAgentBase, "Info": core.Info}
        with redirect_stdout(sys.stderr):
            exec(request["code"], namespace, namespace)
            system.forward = types.MethodType(namespace["forward"], system)
            result = system.forward(core.Info("task", "user", request["question"], None, None, None, -1))
        if not isinstance(result, core.Info):
            raise ValueError("forward must return Info")
        send({"kind": "result", "value": result._asdict()})
    except BaseException as exc:
        send({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
