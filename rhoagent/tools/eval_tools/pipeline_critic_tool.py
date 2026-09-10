import json
import os
from typing import Any, Dict

from rhoagent.llm import get_chat_model
from rhoagent.llm.schema import Message
from rhoagent.tools.base import BaseTool, register_tool


CRITIC_SYSTEM_PROMPT_EN = """
You are a general-purpose "Pipeline Quality Evaluation Critic" for assessing the quality of results from any step in a multi-step task pipeline.

Your input is a JSON string containing:
- step_name: Step name (e.g., "backbone_design", "sequence_design", etc.)
- status: "success" / "failed" / "partial"
- key_metrics: Dictionary containing key metrics for this step (e.g., number generated, pass rate, TM-score, etc.)
- suggestion_for_critic: Points that the main task Agent thinks you should focus on

Your tasks:
1. Strictly based on the information in the JSON, evaluate the quality of results for this step.
   - Key evaluation criteria:
     * TM-score: A TM-score around 0.4 or above is considered acceptable for structure similarity evaluation. 
     * Other metrics: Evaluate based on context-specific thresholds (e.g., sequence recovery rate（sequence recovery rate >= 70% is considered too high to pass）, pass rates, etc.).
2. Provide a clear decision instruction (decision):
   - "ACCEPT_AND_CONTINUE": Results are of sufficient quality to proceed to the next step.
   - "RETRY_RELATED_STEP": Suggest rerunning/redesigning a batch at the current step or related steps, with specific reasons.
   - "ABORT_PIPELINE": Serious problems detected (e.g., environment errors, obviously unreasonable inputs), suggest aborting the current pipeline and requiring manual intervention.
3. Provide a brief reason in natural language (reason), highlighting:
   - Which metrics indicate that results are insufficient to continue;
   - If rerunning is needed, which hyperparameters should be adjusted first (e.g., num_samples, filtering thresholds, etc.).

Output format (must be a JSON string):
{
  "step_name": "<same as input>",
  "decision": "ACCEPT_AND_CONTINUE" | "RETRY_RELATED_STEP" | "ABORT_PIPELINE",
  "reason": "<brief reason in English>",
  "suggested_actions": [
    "<if rerunning is needed, suggest how to change parameters / change process>",
    "<can be empty list>"
  ]
}

Note:
- Do not assume you can directly call tools; you are only a "review" and "decision" component.
- If information is insufficient to make a judgment, clearly state "insufficient information" in reason,
  and in suggested_actions, suggest which additional metrics the main Agent should report.
""".strip()


def _build_llm_cfg_from_env() -> Dict[str, Any]:
    """Build LLM configuration from environment variables.

    Depends on conventions in rhoagent/rhoagent/.env_new:
        DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
    """
    model = os.getenv("DEFAULT_MODEL")
    api_base = os.getenv("DEFAULT_API_BASE")
    api_key = os.getenv("DEFAULT_API_KEY")
    if not (model and api_base and api_key):
        raise RuntimeError(
            "Environment variables incomplete, please first `source rhoagent/rhoagent/.env_new` externally, "
            "at least need DEFAULT_MODEL / DEFAULT_API_BASE / DEFAULT_API_KEY."
        )
    return {
        "model": model,
        "model_server": api_base,
        "api_key": api_key,
    }


def run_pipeline_critic(step_summary: Dict[str, Any]) -> str:
    """Call LLM to provide pipeline decision based on step_summary.
    
    Returns JSON string containing:
    - decision_json: The critic's decision JSON response
    - critic_conversation: Full conversation history with critic agent (for dialogue history)
    """
    llm_cfg = _build_llm_cfg_from_env()
    chat_model = get_chat_model(llm_cfg)

    user_content = json.dumps(step_summary, ensure_ascii=False)
    messages = [
        Message(role="system", content=CRITIC_SYSTEM_PROMPT_EN),
        Message(role="user", content=user_content),
    ]

    # Non-streaming call, directly get final response
    responses = chat_model.chat(messages=messages, stream=False)
    if not responses:
        raise RuntimeError("Critic LLM did not return any content.")
    
    critic_response = responses[-1].content
    
    # Build full conversation history for dialogue history
    critic_conversation = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            critic_conversation.append(msg.model_dump())
        else:
            critic_conversation.append({
                "role": msg.role if hasattr(msg, "role") else "system",
                "content": msg.content if hasattr(msg, "content") else str(msg)
            })
    # Add critic's response
    critic_conversation.append({
        "role": "assistant",
        "content": critic_response
    })
    
    # Return JSON containing both decision and full conversation
    result = {
        "decision_json": critic_response,
        "critic_conversation": critic_conversation,
        "step_summary": step_summary
    }
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@register_tool("critic_agent_tool")
class CriticAgentTool(BaseTool):
    """General Critic Agent tool: Receives JSON summary of a step, returns decision JSON for continue/retry/abort."""

    description = "Perform quality evaluation on JSON summary of any pipeline step and provide `ACCEPT_AND_CONTINUE` / `RETRY_THIS_STEP` / `ABORT_PIPELINE` decision."
    parameters = {
        "type": "object",
        "properties": {
            "step_summary": {
                "oneOf": [
                    {"type": "object"},
                    {"type": "string"}
                ],
                "description": "主 Agent 生成的步骤摘要 JSON（对象或 JSON 字符串），包含 step_name/status/key_metrics/suggestion_for_critic 等字段。",
            }
        },
        "required": ["step_summary"],
    }

    def call(self, params, **kwargs) -> str:
        # If params is a string, parse it first
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                pass  # If not JSON, let _verify_json_format_args handle it
        
        # If step_summary is a string, parse it first (handle double serialization case)
        if isinstance(params, dict) and "step_summary" in params:
            if isinstance(params["step_summary"], str):
                try:
                    params["step_summary"] = json.loads(params["step_summary"])
                except Exception as e:
                    raise ValueError(f"step_summary is not valid JSON: {e}")
        
        args = self._verify_json_format_args(params)
        step_summary = args["step_summary"]
        
        # Check again (prevent it from still being a string after verification)
        if isinstance(step_summary, str):
            try:
                step_summary = json.loads(step_summary)
            except Exception as e:
                raise ValueError(f"step_summary is not valid JSON: {e}")
        if not isinstance(step_summary, dict):
            raise ValueError("step_summary must be a JSON object.")

        decision_json = run_pipeline_critic(step_summary)
        return decision_json


