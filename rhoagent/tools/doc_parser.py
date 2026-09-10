"""
Doc parser tool & compatibility helpers.

- 提供 `DocParser` 类，兼容旧代码 `from rhoagent.tools.doc_parser import DocParser`。
- 提供 `doc_parser` 工具，供 Memory agent 使用（名称与 Memory 中 hard-code 的一致）。
"""

import json
from typing import Any, Dict

from rhoagent.tools.base import BaseTool, register_tool
from rhoagent.tools.simple_doc_parser import SimpleDocParser as DocParser

__all__ = ["DocParser"]


@register_tool("doc_parser")
class DocParserTool(BaseTool):
    """文档解析工具：将指定路径的文件解析为文本片段列表。"""

    description = "解析上传的文档并返回结构化文本，用于后续检索。支持 PDF/Word/PPT/文本等常见格式。"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要解析的文件路径列表。",
            }
        },
        "required": ["paths"],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        paths = args.get("paths", [])
        parser = DocParser()
        results = parser.parse_files(paths)
        return json.dumps(results, ensure_ascii=False)

