# LLM Weighted Filter Analysis Report

Generated at: 2026-01-10 20:26:20

================================================================================
LLM Weighted Filter Analysis
================================================================================
Input CSV: /workspace/hedongchen/RhoAgent/rhoagent/tools/eval_tools
Top-K: 20
Model: Qwen/Qwen3-32B
API Base: https://api.siliconflow.cn/v1

Starting analysis...
Note: This may take some time, as it requires:
  1. Generating alignment descriptions for each candidate (calling USalign)
  2. Generating pseudoknot alignment analysis
  3. Generating motif region secondary structure analysis
  4. Calling LLM to determine weights and filter

Warning: Could not generate example candidate data: [Errno 21] Is a directory: '/workspace/hedongchen/RhoAgent/rhoagent/tools/eval_tools'
## Example Candidate Data JSON (Input to LLM)

Example data generation skipped due to error.

================================================================================
Error Occurred
================================================================================
Error message: [Errno 21] Is a directory: '/workspace/hedongchen/RhoAgent/rhoagent/tools/eval_tools'

Detailed error:
## Error

Error message: [Errno 21] Is a directory: '/workspace/hedongchen/RhoAgent/rhoagent/tools/eval_tools'

### Traceback

```python
Traceback (most recent call last):
  File "/workspace/hedongchen/RhoAgent/tests/run_llm_filter_analysis.py", line 206, in main
    result_json = llm_weighted_filter(
  File "/workspace/hedongchen/RhoAgent/rhoagent/tools/eval_tools/llm_filter_tool.py", line 372, in llm_weighted_filter
    with open(input_path, 'r', encoding='utf-8') as f:
IsADirectoryError: [Errno 21] Is a directory: '/workspace/hedongchen/RhoAgent/rhoagent/tools/eval_tools'

```

