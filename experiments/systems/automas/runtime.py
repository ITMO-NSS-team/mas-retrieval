"""Request admission before a primitive crosses the MCP process boundary."""
from pydantic_ai.mcp import MCPServerStdio


class BudgetedMCPServer(MCPServerStdio):
    def tool_for_tool_def(self, tool_def):
        # retrieve/rerank share state within one agent's stdio server.
        tool_def.sequential = True
        return super().tool_for_tool_def(tool_def)


def tool_hook(session):
    async def call(ctx, call_tool, name, args):
        if name not in {"retrieve", "rerank", "calculate"}:
            raise ValueError(f"Unexpected MCP tool: {name}")
        session.tool()
        query = args.get("query", args.get("expression", ""))
        top_k = args.get("top_k", 20 if name == "retrieve" else 10 if name == "rerank" else 0)
        with session.tracker.track_tool(name, query, top_k) as results:
            value = await call_tool(name, args)
            if name in {"retrieve", "rerank"}:
                if not isinstance(value, dict) or not isinstance(value.get("doc_ids"), list) or not isinstance(value.get("text"), str):
                    raise ValueError("MCP retrieval did not return source IDs and text")
                if not all(isinstance(doc_id, str) for doc_id in value["doc_ids"]):
                    raise ValueError("MCP source IDs must be strings")
                results.extend(value["doc_ids"])
                return value["text"]
            return value
    return call
