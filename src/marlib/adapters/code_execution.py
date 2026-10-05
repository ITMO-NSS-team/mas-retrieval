"""Supervise generated Python and admit its model/tool requests in the parent."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import time
from uuid import uuid4

from marlib.adapters.code_worker import validate_code
from marlib.adapters.tools import do_calculate, do_retrieve, do_rerank
from marlib.tracing.resources import tracked_async_client


class GeneratedCodeError(RuntimeError):
    pass


async def run_generated(*, code, question, core_path, system_config, session, retriever,
                        model, temperature, artifacts):
    validate_code(code)
    if session.limits.wall_seconds is None:
        raise ValueError("Generated-code execution requires wall_seconds")
    remaining = session.limits.wall_seconds - (time.perf_counter() - session.started)
    if remaining <= 0:
        session.stop("wall_seconds")
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    work = artifacts / "worker_cwd"
    work.mkdir(exist_ok=True)
    worker = Path(__file__).with_name("code_worker.py").resolve()
    # -I ignores PYTHONPATH and user site packages. Do not forward credentials,
    # HOME, .env configuration or evaluator data to the generated-code process.
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "SYSTEMROOT") if key in os.environ}
    process = None
    last_retrieved = []
    stderr = (artifacts / "worker.stderr").open("w")
    deadline = asyncio.timeout(remaining)
    try:
        async with deadline:
            process = await asyncio.create_subprocess_exec(sys.executable, "-I", str(worker),
                cwd=work, env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=stderr, start_new_session=True, limit=16 * 1024 * 1024)
            session.write({"kind": "worker_start", "pid": process.pid, "deadline_seconds": remaining})
            payload = {"code": code, "question": question, "core_path": str(Path(core_path).resolve()),
                       "system_config": system_config}
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()
            client = tracked_async_client(session, "answer_execution",
                                           base_url=os.environ.get("OPENAI_BASE_URL"),
                                           api_key=os.environ.get("OPENAI_API_KEY"))
            async with client:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        raise GeneratedCodeError("Worker exited before returning a result; see worker.stderr")
                    request = json.loads(line)
                    kind = request["kind"]
                    if kind == "error":
                        raise GeneratedCodeError(request["error"])
                    if kind == "result":
                        session.write({"kind": "worker_result", "value": request["value"]})
                        return request["value"]
                    if kind == "llm":
                        fields = request["output_fields"]
                        node_temperature = temperature if temperature is not None else request.get("temperature")
                        if node_temperature is not None and (not isinstance(node_temperature, (int, float)) or not 0 <= node_temperature <= 2):
                            raise GeneratedCodeError("Invalid node temperature")
                        result = None
                        logical_id = uuid4().hex
                        for _ in range(5):
                            response = await client.chat.completions.create(
                                model=model, messages=request["messages"],
                                **({"temperature": node_temperature} if node_temperature is not None else {}),
                                _marlib_logical_call_id=logical_id,
                                response_format={"type": "json_object"})
                            try:
                                value = json.loads(response.choices[0].message.content or "")
                                if isinstance(value, dict) and set(fields) <= value.keys():
                                    result = {key: value[key] for key in fields}
                                    break
                            except (ValueError, TypeError):
                                pass
                        if result is None:
                            raise GeneratedCodeError("Node did not produce required JSON fields in five attempts")
                    elif kind in {"retrieve", "rerank", "calculate"}:
                        session.tool()
                        query = request.get("query", request.get("expression", ""))
                        top_k = request.get("top_k", 0)
                        with session.tracker.track_tool(kind, query, top_k) as ids:
                            if kind == "retrieve":
                                last_retrieved, result = await asyncio.to_thread(do_retrieve, retriever, query, top_k)
                                ids.extend(d.doc_id for d in last_retrieved)
                            elif kind == "rerank":
                                last_retrieved, result = await asyncio.to_thread(do_rerank, retriever, query, last_retrieved, top_k)
                                ids.extend(d.doc_id for d in last_retrieved)
                            else:
                                result = do_calculate(query)
                    else:
                        raise GeneratedCodeError(f"Unknown worker operation: {kind}")
                    process.stdin.write((json.dumps({"result": result}) + "\n").encode())
                    await process.stdin.drain()
    except TimeoutError:
        if deadline.expired():
            session.stop("wall_seconds")
        raise
    finally:
        if process:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await process.wait()
            session.write({"kind": "worker_end", "pid": process.pid, "returncode": process.returncode})
        stderr.close()
