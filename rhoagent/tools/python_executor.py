"""
A simple Python code execution tool.

This module provides a `PythonExecutor` tool so that agents (e.g., TIRMathAgent)
can execute small Python snippets. It is intentionally minimal and sandboxed:
- No external imports are added automatically.
- Code runs in an isolated namespace.
"""

import io
import json
from contextlib import redirect_stdout
from typing import Any, Dict

from rhoagent.tools.base import BaseTool, register_tool


@register_tool("python_executor")
class PythonExecutor(BaseTool):
    """Execute short Python code snippets and return stdout + last expression value."""

    description = "执行一小段 Python 代码，并返回标准输出和最后一个表达式的值。仅用于轻量级计算与调试。"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码。可以包含多行；如需要返回值，请在最后一行写表达式。",
            }
        },
        "required": ["code"],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        code: str = args["code"]

        # 独立命名空间，避免污染全局
        local_ns: Dict[str, Any] = {}
        stdout_buf = io.StringIO()

        # 为了支持“最后一行表达式”的返回，我们拆分代码
        lines = [ln for ln in code.splitlines() if ln.strip() != ""]
        if not lines:
            return json.dumps({"success": False, "error": "Empty code."}, ensure_ascii=False)

        *body_lines, last_line = lines
        body_code = "\n".join(body_lines)
        last_expr = last_line

        try:
            with redirect_stdout(stdout_buf):
                if body_code.strip():
                    exec(body_code, {}, local_ns)  # noqa: S102
                # 尝试把最后一行当作表达式求值
                try:
                    result = eval(last_expr, {}, local_ns)  # noqa: S307
                except SyntaxError:
                    # 不是表达式，当作普通语句执行
                    exec(last_expr, {}, local_ns)  # noqa: S102
                    result = None

            stdout = stdout_buf.getvalue()
            return json.dumps(
                {
                    "success": True,
                    "stdout": stdout,
                    "result": result,
                },
                ensure_ascii=False,
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "stdout": stdout_buf.getvalue(),
                },
                ensure_ascii=False,
            )




