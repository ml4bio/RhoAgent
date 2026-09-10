"""
Minimal retrieval tool for RAG compatibility.

This is a lightweight implementation to satisfy the `retrieval` tool name
expected by `Memory`. It simply echoes back basic info about the files and
query, without performing real vector search.
"""

import json
from typing import Any, Dict, List

from rhoagent.tools.base import BaseTool, register_tool


@register_tool("retrieval")
class RetrievalTool(BaseTool):
    """Dummy retrieval tool that returns file list and query for debugging/compat."""

    description = "检索工具占位实现：返回查询文本以及可用文件的基本信息，供调试和兼容使用。"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用户查询文本。",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可用于检索的文件路径列表。",
            },
        },
        "required": ["query", "files"],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        query: str = args.get("query", "")
        files: List[str] = args.get("files", [])
        return json.dumps(
            {
                "success": True,
                "query": query,
                "num_files": len(files),
                "files": files,
                "note": "This is a minimal retrieval placeholder; no real semantic search is performed.",
            },
            ensure_ascii=False,
            indent=2,
        )




