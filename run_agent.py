import os
import json
from typing import Any, Dict, List

from rhoagent.agents.rhoagent_pipeline import build_rna_design_pipeline_agent
from rhoagent.utils.output_beautify import typewriter_print


def build_llm_cfg_from_env() -> Dict[str, Any]:
    """Build LLM configuration from environment variables (depends on conventions in rhoagent/.env_new)."""
    model = os.getenv("DEFAULT_MODEL")
    api_base = os.getenv("DEFAULT_API_BASE")
    api_key = os.getenv("DEFAULT_API_KEY")
    if not (model and api_base and api_key):
        raise RuntimeError(
            "Environment variables are incomplete. Please `source rhoagent/.env_new` first. "
            "At least DEFAULT_MODEL / DEFAULT_API_BASE / DEFAULT_API_KEY are required."
        )
    return {
        "model": model,
        "model_server": api_base,
        "api_key": api_key,
    }


def serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert Message / dict to serializable list."""
    serialized = []
    for m in messages:
        if hasattr(m, "model_dump"):
            serialized.append(m.model_dump())
        else:
            serialized.append(m)
    return serialized


def save_history(messages: List[Any], history_path: str):
    """Save conversation history to specified path (overwrite)."""
    if not history_path:
        return
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serialize_messages(messages), f, ensure_ascii=False, indent=2)


def extract_critic_conversations(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract critic agent conversations from agent messages and add them to history."""
    critic_conversations = []
    
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            # Check if message contains critic agent tool call result
            if isinstance(content, str) and "critic_conversation" in content:
                try:
                    # Try to find and parse JSON objects containing critic_conversation
                    import re
                    # Find JSON objects that might contain critic_conversation (more flexible pattern)
                    # Look for opening brace followed by content that includes critic_conversation
                    start_idx = content.find('"critic_conversation"')
                    if start_idx != -1:
                        # Find the opening brace before this
                        brace_start = content.rfind('{', 0, start_idx)
                        if brace_start != -1:
                            # Try to find the matching closing brace
                            brace_count = 0
                            brace_end = -1
                            for i in range(brace_start, len(content)):
                                if content[i] == '{':
                                    brace_count += 1
                                elif content[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        brace_end = i + 1
                                        break
                            
                            if brace_end > brace_start:
                                json_str = content[brace_start:brace_end]
                                try:
                                    parsed = json.loads(json_str)
                                    if "critic_conversation" in parsed and isinstance(parsed["critic_conversation"], list):
                                        critic_conversations.extend(parsed["critic_conversation"])
                                except json.JSONDecodeError:
                                    pass
                except Exception:
                    pass
    
    return critic_conversations


def run_chat(agent, messages: List[Dict[str, str]], history_path: str):
    """Simple interactive loop: stream agent responses, allow user to provide additional information in multiple rounds, and continuously save history.
    Critic agent conversations are also saved to dialogue history."""
    response_plain_text = ""
    while True:
        # Agent response (streaming)
        last_chunk = []
        for last_chunk in agent.run(messages=messages):
            response_plain_text = typewriter_print(last_chunk, response_plain_text)
        if last_chunk:
            messages.extend(last_chunk)
            
            # Extract and add critic agent conversations to history
            critic_convs = extract_critic_conversations(last_chunk)
            if critic_convs:
                # Add a separator message before critic conversations
                messages.append({
                    "role": "system",
                    "content": "[Critic Agent Conversation]",
                    "_source": "critic_agent_separator"
                })
                # Add critic conversations with a marker to distinguish them
                for conv in critic_convs:
                    if isinstance(conv, dict) and "role" in conv:
                        conv_with_marker = {
                            **conv,
                            "_source": "critic_agent"
                        }
                        messages.append(conv_with_marker)
            
            save_history(messages, history_path)

        # User continues input
        user_input = input("\n[user] Continue input (press Enter to exit): ").strip()
        if not user_input:
            break
        messages.append({"role": "user", "content": user_input})
        save_history(messages, history_path)
        print("\n==== Agent Streaming Output ====\n")


def main():
    """Use RNA Design Pipeline Agent only, agent plans and calls tools to complete each step (interactive)."""
    llm_cfg = build_llm_cfg_from_env()
    agent = build_rna_design_pipeline_agent(llm=llm_cfg)

    print("==== RNA Design Pipeline Agent (Interactive Example) ====")
    print("Tip: Press Enter directly to use default example parameters; enter custom content to set your own parameters.")

    first_input = input(
        "\nEnter your first message (press Enter directly to use default example parameters and start):\n> "
    ).strip()

    history_path = os.path.join(
        default_work_dir,
        "agent_dialog_history.json",
    )

    if first_input:
        messages = [{"role": "user", "content": first_input}]
    else:
        # Default to Template 1: user has provided target_ss + fixed_motif + wildtype_seq
        default_wildtype = (
            "ACACCCACCUGUUACAAGUCAGGACAGAAGCAGAGUAACGGUUGCUUACGCAACCGGUAAUGCUACUGGGUGU"
        )
        # Default example parameters (consistent with tests/test_twister.py), user can press Enter to use directly
        default_target_ss = "(((((((((((.[[[..{{)))).....((((..]]].(((((((....)))))))....))))}})))))))"
        default_fixed_motif = "-----------UUACAA---------------G-GUA--------------------UAA-------------"
        default_work_dir = os.path.join(
            os.path.dirname(__file__),
            "tests",
            "design_example",
        )
        default_user_query = f"""
I want to design a functional RNA structure with the following parameters:
- wildtype_seq: {default_wildtype}
- target_ss: {default_target_ss}
- fixed_motif: {default_fixed_motif}
- work_dir: {os.path.abspath(default_work_dir)}
"""
        messages = [{"role": "user", "content": default_user_query.strip()}]
    save_history(messages, history_path)

    print("\n==== Agent Streaming Output ====\n")
    run_chat(agent, messages, history_path)


if __name__ == "__main__":
    main()


