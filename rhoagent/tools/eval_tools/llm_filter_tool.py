from __future__ import annotations

import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from rhoagent.llm import get_chat_model
from rhoagent.llm.schema import Message, SYSTEM, USER
from rhoagent.tools.base import BaseTool, register_tool

from .utils.eval_utils import DEFAULT_USALIGN_BIN, extract_fixed_motif_indices
from .utils.pdb_utils import extract_rna_sequence_to_fasta
from .utils.usalign_to_nl import usalign_to_nl


def describe_pseudoknot_alignment(pred_ss: str, target_ss: str) -> str:
    """
    对 target_ss 的伪结区域，检查 pred_ss 是否有对应的，生成自然语言描述。
    """
    if len(pred_ss) != len(target_ss):
        return "Error: pred_ss and target_ss have different lengths."
    
    def _pk_positions(s: str) -> set:
        return {i for i, ch in enumerate(s) if ch in "[]{}"}
    
    target_pk = _pk_positions(target_ss)
    pred_pk = _pk_positions(pred_ss)
    
    if not target_pk:
        return "Target structure has no pseudoknot regions."
    
    # 找出伪结对
    def _extract_pk_pairs(ss: str) -> List[tuple]:
        pairs = []
        stack_square = []
        stack_curly = []
        for i, char in enumerate(ss):
            if char == '[':
                stack_square.append(i)
            elif char == ']':
                if stack_square:
                    j = stack_square.pop()
                    pairs.append((j, i, '[]'))
            elif char == '{':
                stack_curly.append(i)
            elif char == '}':
                if stack_curly:
                    j = stack_curly.pop()
                    pairs.append((j, i, '{}'))
        return pairs
    
    target_pairs = _extract_pk_pairs(target_ss)
    pred_pairs = _extract_pk_pairs(pred_ss)
    
    # 检查每个 target 伪结对是否在 pred 中存在
    matched_pairs = []
    unmatched_pairs = []
    
    for t_start, t_end, t_type in target_pairs:
        found = False
        for p_start, p_end, p_type in pred_pairs:
            if t_start == p_start and t_end == p_end:
                matched_pairs.append((t_start, t_end, t_type))
                found = True
                break
        if not found:
            unmatched_pairs.append((t_start, t_end, t_type))
    
    tp = len(matched_pairs)
    fn = len(unmatched_pairs)
    total = len(target_pairs)
    
    description_parts = []
    if total > 0:
        recall = tp / total if total > 0 else 0.0
        description_parts.append(f"Target structure has {total} pseudoknot pair(s): {len([p for p in target_pairs if p[2] == '[]'])} square bracket pair(s) and {len([p for p in target_pairs if p[2] == '{}'])} curly bracket pair(s).")
        description_parts.append(f"Predicted structure correctly identifies {tp} out of {total} pseudoknot pair(s) (recall: {recall:.2%}).")
        
        if matched_pairs:
            matched_str = ", ".join([f"positions {s}-{e} ({t})" for s, e, t in matched_pairs[:3]])
            if len(matched_pairs) > 3:
                matched_str += f" and {len(matched_pairs) - 3} more"
            description_parts.append(f"Matched pairs: {matched_str}.")
        
        if unmatched_pairs:
            unmatched_str = ", ".join([f"positions {s}-{e} ({t})" for s, e, t in unmatched_pairs[:3]])
            if len(unmatched_pairs) > 3:
                unmatched_str += f" and {len(unmatched_pairs) - 3} more"
            description_parts.append(f"Missing pairs in prediction: {unmatched_str}.")
    else:
        description_parts.append("Target structure has no pseudoknot regions.")
    
    return " ".join(description_parts)


def describe_motif_ss_alignment(pred_ss: str, target_ss: str, fixed_motif: str) -> str:
    """
    根据 fixed_motif，对 motif 区域的二级结构，检查 pred_ss 是否跟 target_ss 对齐，生成自然语言描述。
    """
    if len(pred_ss) != len(target_ss) or len(pred_ss) != len(fixed_motif):
        return "Error: pred_ss, target_ss, and fixed_motif have different lengths."
    
    motif_indices = extract_fixed_motif_indices(fixed_motif)
    if not motif_indices:
        return "No fixed motif positions found."
    
    # 提取 motif 区域的字符
    motif_target_ss = ''.join([target_ss[i] if i < len(target_ss) else '' for i in sorted(motif_indices)])
    motif_pred_ss = ''.join([pred_ss[i] if i < len(pred_ss) else '' for i in sorted(motif_indices)])
    
    # 统计匹配情况
    matches = 0
    mismatches = []
    for idx in sorted(motif_indices):
        if idx < len(pred_ss) and idx < len(target_ss):
            if pred_ss[idx] == target_ss[idx]:
                matches += 1
            else:
                mismatches.append((idx, target_ss[idx], pred_ss[idx]))
    
    total_positions = len(motif_indices)
    match_rate = matches / total_positions if total_positions > 0 else 0.0
    
    description_parts = []
    description_parts.append(f"Fixed motif region contains {total_positions} positions.")
    description_parts.append(f"Secondary structure symbols match at {matches} out of {total_positions} positions ({match_rate:.2%}).")
    
    if mismatches:
        mismatch_examples = mismatches[:5]
        mismatch_str = ", ".join([f"position {idx}: target='{t}' vs pred='{p}'" for idx, t, p in mismatch_examples])
        if len(mismatches) > 5:
            mismatch_str += f" and {len(mismatches) - 5} more mismatches"
        description_parts.append(f"Mismatches: {mismatch_str}.")
    else:
        description_parts.append("All motif positions have matching secondary structure symbols.")
    
    # 描述 motif 区域的整体结构特征
    target_chars = {ch: motif_target_ss.count(ch) for ch in set(motif_target_ss)}
    pred_chars = {ch: motif_pred_ss.count(ch) for ch in set(motif_pred_ss)}
    
    target_structure = "Target motif region structure: " + ", ".join([f"{count} '{ch}'" for ch, count in sorted(target_chars.items())])
    pred_structure = "Predicted motif region structure: " + ", ".join([f"{count} '{ch}'" for ch, count in sorted(pred_chars.items())])
    
    description_parts.append(target_structure + ". " + pred_structure + ".")
    
    return " ".join(description_parts)


def _evaluate_single_candidate(
    candidate_data: Dict,
    index: int,
    model_name: str,
    api_base: str,
    api_key: str,
) -> Tuple[int, Dict]:
    """
    对单条候选数据进行 LLM 评估，返回打分（0-10）和评估原因。
    
    Args:
        candidate_data: 候选数据字典，包含所有指标和自然语言描述
        index: 候选数据索引
        model_name: LLM 模型名称
        api_base: API 基础 URL
        api_key: API 密钥
    
    Returns:
        (index, evaluation_dict) 元组，evaluation_dict 包含 score 和 reason
    """
    system_prompt = (
        "You are an RNA structure evaluation expert. Evaluate a single candidate structure "
        "based on multiple metrics and natural language descriptions.\n\n"
        "Key Metrics (higher is better):\n"
        "- tm_score: Structural similarity (0-1), higher indicates better overall structural match\n"
        "- ss_f1score: Secondary structure F1-score (0-1), higher indicates better secondary structure prediction\n"
        "- functional_core_f1score: Functional core region F1-score (0-1), higher indicates better functional core structure\n"
        "- pseudoknot_recall: Pseudoknot recall (0-1), higher indicates better pseudoknot prediction\n"
        "- ss_pairs_ratio: Ratio of canonical pairs in secondary structure (0-1), higher indicates more canonical base pairs\n"
        "- pseudoknot_ss_pairs_ratio: Ratio of canonical pairs in pseudoknot regions (0-1), higher indicates more canonical pairs in pseudoknots\n"
        "- tmscore_wt: TM-score compared to wild-type structure (0-1), higher indicates better similarity to wild-type structure\n"
        "- functional_core_rmsd_norm: Normalized functional core RMSD after Kabsch alignment (0-1), HIGHER is BETTER. "
        "This is the normalized (and inverted) version of the functional core RMSD. "
        "The original RMSD (in Angstroms) measures structural deviation - smaller RMSD values indicate better structural accuracy. "
        "After normalization and inversion, values closer to 1.0 indicate excellent structural accuracy (< 2.0 Å), "
        "values around 0.7-0.9 indicate good accuracy (2.0-5.0 Å), values around 0.4-0.7 indicate moderate accuracy (5.0-10.0 Å), "
        "and values < 0.4 indicate poor accuracy (> 10.0 Å).\n\n"
        "Secondary Metrics (lower is better, indicating more innovation):\n"
        "- seq_rec_rate: Sequence recovery rate (0-1), lower indicates more sequence innovation\n"
        "- seq_similarity: Sequence similarity (0-1), lower indicates more sequence diversity\n\n"
        "You will also receive natural language descriptions:\n"
        "- alignment_description: Structural alignment analysis\n"
        "- pseudoknot_alignment_description: Pseudoknot region analysis\n"
        "- motif_ss_alignment_description: Motif region secondary structure analysis\n\n"
        "Please evaluate this candidate comprehensively and provide:\n"
        "1. A score from 0-10 (10 being best)\n"
        "2. A detailed reason explaining your evaluation\n\n"
        "Return JSON format: {\"score\": int (0-10), \"reason\": string}"
    )
    
    # 构建用户提示
    user_prompt = f"Candidate #{index}:\n\n"
    user_prompt += f"Prediction name: {candidate_data.get('pred_name', 'N/A')}\n\n"
    
    # 添加自然语言描述
    if candidate_data.get('alignment_description'):
        user_prompt += f"Alignment Description:\n{candidate_data['alignment_description']}\n\n"
    if candidate_data.get('pseudoknot_alignment_description'):
        user_prompt += f"Pseudoknot Analysis:\n{candidate_data['pseudoknot_alignment_description']}\n\n"
    if candidate_data.get('motif_ss_alignment_description'):
        user_prompt += f"Motif Region Analysis:\n{candidate_data['motif_ss_alignment_description']}\n\n"
    
    # 添加数值指标
    user_prompt += "Metrics:\n"
    if candidate_data.get('tm_score') is not None:
        user_prompt += f"- tm_score: {candidate_data['tm_score']:.4f}\n"
    if candidate_data.get('ss_f1score') is not None:
        user_prompt += f"- ss_f1score: {candidate_data['ss_f1score']:.4f}\n"
    if candidate_data.get('functional_core_f1score') is not None:
        user_prompt += f"- functional_core_f1score: {candidate_data['functional_core_f1score']:.4f}\n"
    if candidate_data.get('pseudoknot_recall') is not None:
        user_prompt += f"- pseudoknot_recall: {candidate_data['pseudoknot_recall']:.4f}\n"
    if candidate_data.get('tmscore_wt') is not None:
        user_prompt += f"- tmscore_wt: {candidate_data['tmscore_wt']:.4f} (higher is better, similarity to wild-type)\n"
    if candidate_data.get('functional_core_rmsd_norm') is not None:
        rmsd_norm_value = candidate_data['functional_core_rmsd_norm']
        user_prompt += f"- functional_core_rmsd_norm: {rmsd_norm_value:.4f} (higher is better, normalized functional core structural accuracy)\n"
    if candidate_data.get('seq_rec_rate') is not None:
        user_prompt += f"- seq_rec_rate: {candidate_data['seq_rec_rate']:.4f} (lower is better)\n"
    if candidate_data.get('seq_similarity') is not None:
        user_prompt += f"- seq_similarity: {candidate_data['seq_similarity']:.4f} (lower is better)\n"
    if candidate_data.get('ss_pairs_ratio') is not None:
        user_prompt += f"- ss_pairs_ratio: {candidate_data['ss_pairs_ratio']:.4f}\n"
    if candidate_data.get('pseudoknot_ss_pairs_ratio') is not None:
        user_prompt += f"- pseudoknot_ss_pairs_ratio: {candidate_data['pseudoknot_ss_pairs_ratio']:.4f}\n"
    
    user_prompt += "\nPlease provide your evaluation (score 0-10 and detailed reason)."
    
    try:
        # 构建 LLM 配置
        llm_cfg = {'model': model_name}
        if api_base:
            llm_cfg['model_server'] = api_base
            llm_cfg['model_type'] = 'oai'
        if api_key:
            llm_cfg['api_key'] = api_key
        
        llm = get_chat_model(llm_cfg)
        *_, responses = llm.chat(messages=[
            Message(role=SYSTEM, content=system_prompt),
            Message(role=USER, content=user_prompt),
        ])
        content = responses[-1].content if responses else ""
        
        # 提取 JSON
        json_content = content.strip()
        if '```json' in json_content:
            json_content = json_content.split('```json')[1].split('```')[0].strip()
        elif '```' in json_content:
            parts = json_content.split('```')
            if len(parts) >= 3:
                json_content = parts[1].strip()
                if json_content.startswith('json'):
                    json_content = json_content[4:].strip()
        
        if not json_content.startswith('{'):
            start_idx = json_content.find('{')
            end_idx = json_content.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = json_content[start_idx:end_idx+1]
        
        result = json.loads(json_content)
        score = int(result.get('score', 5))
        reason = result.get('reason', 'No reason provided')
        
        # 确保分数在 0-10 范围内
        score = max(0, min(10, score))
        
        return (index, {'score': score, 'reason': reason})
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[ERROR] LLM evaluation failed for candidate #{index}: {error_msg}", file=sys.stderr)
        # 返回默认分数和错误原因
        return (index, {
            'score': 5,
            'reason': f"Evaluation failed: {error_msg[:100]}"
        })


@register_tool('llm_rank_alignments_v2')
class LLMRankAlignmentsTool(BaseTool):
    """
    Call USalign + LLM to score motif alignment descriptions and rank sequences.
    Steps:
    1) For each {pred_path, gt_path}, run usalign_to_nl to get motif_alignment_desc, TM/RMSD.
    2) Extract first RNA chain sequence from pred_path (if any).
    3) Ask LLM to score each candidate 1-10 and return top_k.
    4) Write selected sequences and scores to CSV.
    """

    description = 'Run USalign + LLM scoring for paired PDB lists, return top-k sequences with scores to CSV.'
    parameters = {
        'type': 'object',
        'properties': {
            'pred_paths': {
                'type': 'array',
                'description': 'List of prediction PDB/CIF paths.',
                'items': {'type': 'string'},
            },
            'gt_paths': {
                'type': 'array',
                'description': 'List of ground-truth PDB/CIF paths (same length as pred_paths).',
                'items': {'type': 'string'},
            },
            'motif_template': {
                'type': 'string',
                'description': 'Single motif template string applied to all pairs.',
                'default': '',
            },
            'top_k': {
                'type': 'integer',
                'default': 10,
                'description': 'Return top-k by LLM score.',
            },
            'usalign_bin': {
                'type': 'string',
                'default': DEFAULT_USALIGN_BIN,
                'description': 'Path to USalign binary.',
            },
            'model': {
                'type': 'string',
                'default': 'qwen-plus',
                'description': 'LLM model name for scoring.',
            },
            'output_csv': {
                'type': 'string',
                'description': 'CSV output path (default: ./llm_rank_results.csv).',
            },
        },
        'required': ['pred_paths', 'gt_paths'],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        pred_paths: List[str] = args['pred_paths']
        gt_paths: List[str] = args['gt_paths']
        motif_template: str = args.get('motif_template', '')
        top_k: int = args.get('top_k', 10)
        usalign_bin: str = args.get('usalign_bin', DEFAULT_USALIGN_BIN)
        
        # Load config from environment variables
        model_name = args.get('model') or os.getenv('DEFAULT_MODEL', 'qwen-plus')
        api_base = os.getenv('DEFAULT_API_BASE', '')
        api_key = os.getenv('DEFAULT_API_KEY', '')
        
        output_csv = args.get('output_csv') or str(Path.cwd() / 'llm_rank_results.csv')

        pair_count = min(len(pred_paths), len(gt_paths))
        summaries = []
        for idx in range(pair_count):
            pred_path = pred_paths[idx]
            gt_path = gt_paths[idx]

            result = usalign_to_nl(
                gt_path=gt_path,
                pred_path=pred_path,
                motif_template=motif_template,
                usalign_bin=usalign_bin,
            )

            seq = ''
            seq_extracted = extract_rna_sequence_to_fasta(pred_path)
            if isinstance(seq_extracted, list) and seq_extracted:
                seq = seq_extracted[0]

            summaries.append({
                'pred_path': pred_path,
                'gt_path': gt_path,
                'sequence': seq,
                'motif_alignment_desc': result.get('motif_alignment_desc', ''),
                'tm_score': result.get('tm_score'),
                'rmsd': result.get('rmsd'),
                'alignment_block': result.get('alignment_block'),
                'motif_template': motif_template,
            })

        prompt_items = []
        for idx, item in enumerate(summaries, 1):
            prompt_items.append(
                f"[#{idx}] pred: {item['pred_path']}\n"
                f"gt: {item['gt_path']}\n"
                f"sequence: {item['sequence']}\n"
                f"alignment: {item['motif_alignment_desc']}\n"
                f"tm_score: {item['tm_score']}, rmsd: {item['rmsd']}"
            )

        system_prompt = (
            "You are an RNA structure reviewer. Score each candidate from 1-10 based on alignment quality "
            "(higher TM-score, lower RMSD, better motif alignment). "
            "Return JSON array of objects: {\"index\": int, \"score\": int (1-10), \"reason\": string}. "
            "The reason should briefly justify the recommendation "
            "Be concise."
        )
        user_prompt = "Candidates:\n\n" + "\n\n".join(prompt_items)

        try:
            # Build LLM config from .env_new settings
            llm_cfg = {'model': model_name}
            if api_base:
                llm_cfg['model_server'] = api_base
                # Explicitly set model_type to 'oai' for OpenAI-compatible APIs
                llm_cfg['model_type'] = 'oai'
            if api_key:
                llm_cfg['api_key'] = api_key
            
            llm = get_chat_model(llm_cfg)
            *_, responses = llm.chat(messages=[
                Message(role=SYSTEM, content=system_prompt),
                Message(role=USER, content=user_prompt),
            ])
            content = responses[0].content if responses else ""
            
            # Try to extract JSON from response (might be wrapped in markdown code blocks)
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            parsed = json.loads(content)
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"[ERROR] LLM call failed: {error_msg}", file=sys.stderr)
            print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
            parsed = [{"index": i + 1, "score": int((item['tm_score'] or 0) * 10),
                       "reason": f"heuristic by TM-score (fallback: {error_msg[:50]})"} for i, item in enumerate(summaries)]

        scores = {}
        for obj in parsed:
            idx = obj.get('index')
            if idx is None:
                continue
            scores[int(idx)] = {'score': obj.get('score'), 'reason': obj.get('reason', '')}

        ranked = []
        for i, item in enumerate(summaries, 1):
            sc = scores.get(i, {})
            ranked.append({**item, 'score': sc.get('score'), 'reason': sc.get('reason')})

        ranked = sorted(ranked, key=lambda x: (x['score'] is not None, x['score']), reverse=True)
        top_k = max(1, min(top_k, len(ranked)))
        top_list = ranked[:top_k]

        fieldnames = ['pred_path', 'gt_path', 'sequence', 'score', 'reason', 'tm_score', 'rmsd', 'motif_alignment_desc']
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in top_list:
                writer.writerow({k: row.get(k) for k in fieldnames})

        return json.dumps({
            "success": True,
            "model": model_name,
            "top_k": top_k,
            "results": top_list,
            "output_csv": output_csv,
        }, ensure_ascii=False)


def llm_weighted_filter(
    input_csv: str,
    output_csv: str = None,
    top_k: int = 20,
    model: str = None,
    weight_tm_score: float = None,
    weight_seq_rec_rate: float = None,
    weight_reason: str = None,
    usalign_bin: str = DEFAULT_USALIGN_BIN,
) -> str:
    """
    Use two-stage LLM evaluation to filter top-k records from first-round filtered CSV.
    
    Workflow:
    1. Stage 1 - Parallel LLM Evaluation: For each candidate, LLM evaluates based on metrics and natural 
       language descriptions, providing a score (0-10) and evaluation reason. Up to 20 evaluations run in parallel.
    2. Stage 2 - Weight Determination: All evaluation results are analyzed by LLM to determine optimal weights 
       for different metrics based on correlation with LLM scores.
    3. Stage 3 - Composite Scoring: Calculate weighted composite scores using determined weights and select top-k.
    
    Enhanced version includes alignment descriptions, pseudoknot analysis, and motif region analysis.
    
    Args:
        input_csv: Input filtered CSV file path (from first_round_filter_tool output)
        output_csv: Output final filtered CSV file path, auto-generated if None
        top_k: Return top-k records, default 20
        model: LLM model name, read from environment variables if None
        weight_tm_score: Externally provided TM-score weight (optional, if provided skips LLM evaluation and uses external weights)
        weight_seq_rec_rate: Externally provided seq_rec_rate weight (optional, if provided skips LLM evaluation and uses external weights)
        weight_reason: Reason for weight selection (optional, used with external weights)
        usalign_bin: Path to USalign binary
    
    Returns:
        JSON-formatted result summary including LLM evaluation scores and reasons for each candidate
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        return json.dumps({
            "success": False,
            "error": f"Input file does not exist: {input_csv}",
        }, ensure_ascii=False)
    
    # Read CSV file
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    if not rows:
        return json.dumps({
            "success": False,
            "error": "CSV file is empty",
        }, ensure_ascii=False)
    
    # Process each row to generate natural language descriptions
    enriched_rows = []
    
    for idx, row in enumerate(rows):
        enriched_row = dict(row)
        
        # Get paths
        pred_path = row.get('pred_path', '') or row.get('pred_file', '')
        gt_path = row.get('gt_path', '') or row.get('gt_file', '')
        
        # Generate alignment description using usalign_to_nl
        alignment_desc = ""
        if pred_path and gt_path and Path(pred_path).exists() and Path(gt_path).exists():
            try:
                fixed_motif = row.get('fixed_motif', '')
                result = usalign_to_nl(
                    gt_path=gt_path,
                    pred_path=pred_path,
                    motif_template=fixed_motif,
                    usalign_bin=usalign_bin,
                )
                alignment_desc = result.get('motif_alignment_desc', '')
            except Exception as e:
                alignment_desc = f"Error generating alignment description: {str(e)}"
        else:
            alignment_desc = "Alignment description not available (missing paths or files)."
        
        # Generate pseudoknot alignment description
        pseudoknot_desc = ""
        pred_ss = row.get('pred_ss', '')
        target_ss = row.get('target_ss', '')
        if pred_ss and target_ss:
            pseudoknot_desc = describe_pseudoknot_alignment(pred_ss, target_ss)
        else:
            pseudoknot_desc = "Pseudoknot analysis not available (missing pred_ss or target_ss)."
        
        # Generate motif region SS alignment description
        motif_desc = ""
        fixed_motif = row.get('fixed_motif', '')
        if pred_ss and target_ss and fixed_motif:
            motif_desc = describe_motif_ss_alignment(pred_ss, target_ss, fixed_motif)
        else:
            motif_desc = "Motif region analysis not available (missing pred_ss, target_ss, or fixed_motif)."
        
        enriched_row['alignment_description'] = alignment_desc
        enriched_row['pseudoknot_alignment_description'] = pseudoknot_desc
        enriched_row['motif_ss_alignment_description'] = motif_desc
        
        enriched_rows.append(enriched_row)
    
    # Prepare statistics for LLM reference
    try:
        tm_scores = [float(row.get('tm_score', 0)) for row in enriched_rows if row.get('tm_score')]
        seq_rec_rates = [float(row.get('seq_rec_rate', 0)) for row in enriched_rows if row.get('seq_rec_rate')]
        seq_similarities = [float(row.get('seq_similarity', 0)) for row in enriched_rows if row.get('seq_similarity')]
        ss_f1scores = [float(row.get('ss_f1score', 0)) for row in enriched_rows if row.get('ss_f1score')]
        functional_core_f1scores = [float(row.get('functional_core_f1score', 0)) for row in enriched_rows if row.get('functional_core_f1score')]
        pseudoknot_recalls = [float(row.get('pseudoknot_recall', 0)) for row in enriched_rows if row.get('pseudoknot_recall')]
        
        stats = {
            'total_count': len(enriched_rows),
            'tm_score_range': [min(tm_scores), max(tm_scores)] if tm_scores else None,
            'tm_score_avg': sum(tm_scores) / len(tm_scores) if tm_scores else None,
            'seq_rec_rate_range': [min(seq_rec_rates), max(seq_rec_rates)] if seq_rec_rates else None,
            'seq_rec_rate_avg': sum(seq_rec_rates) / len(seq_rec_rates) if seq_rec_rates else None,
            'seq_similarity_range': [min(seq_similarities), max(seq_similarities)] if seq_similarities else None,
            'seq_similarity_avg': sum(seq_similarities) / len(seq_similarities) if seq_similarities else None,
            'ss_f1score_range': [min(ss_f1scores), max(ss_f1scores)] if ss_f1scores else None,
            'ss_f1score_avg': sum(ss_f1scores) / len(ss_f1scores) if ss_f1scores else None,
            'functional_core_f1score_range': [min(functional_core_f1scores), max(functional_core_f1scores)] if functional_core_f1scores else None,
            'functional_core_f1score_avg': sum(functional_core_f1scores) / len(functional_core_f1scores) if functional_core_f1scores else None,
            'pseudoknot_recall_range': [min(pseudoknot_recalls), max(pseudoknot_recalls)] if pseudoknot_recalls else None,
            'pseudoknot_recall_avg': sum(pseudoknot_recalls) / len(pseudoknot_recalls) if pseudoknot_recalls else None,
        }
    except (ValueError, KeyError):
        stats = {'total_count': len(enriched_rows)}
    
    # Check if external weights are provided (for backward compatibility, if only tm_score and seq_rec_rate are provided)
    if weight_tm_score is not None and weight_seq_rec_rate is not None:
        # Use externally provided weights (backward compatibility mode)
        weight_tm = float(weight_tm_score)
        weight_seq_rec = float(weight_seq_rec_rate)
        weight_reason = weight_reason or "Using externally provided weights (from wet_lab_analysis_tool)"
        
        # Ensure weights are within reasonable range and normalize
        weight_tm = max(0.0, min(1.0, weight_tm))
        weight_seq_rec = max(0.0, min(1.0, weight_seq_rec))
        total_weight = weight_tm + weight_seq_rec
        if total_weight > 0:
            weight_tm = weight_tm / total_weight
            weight_seq_rec = weight_seq_rec / total_weight
        else:
            # If all weights are 0, use default values
            weight_tm = 0.65
            weight_seq_rec = 0.35
        
        # Set other weights to 0 for backward compatibility
        weight_ss_f1 = 0.0
        weight_func_core = 0.0
        weight_pk_recall = 0.0
        weight_tm_wt = 0.0
        weight_func_rmsd_norm = 0.0
        weight_seq_sim = 0.0
        weight_ss_pairs = 0.0
        weight_pk_pairs = 0.0
    else:
        # Step 1: Prepare candidate data and parallel LLM evaluation
        model_name = model or os.getenv('DEFAULT_MODEL', 'qwen-plus')
        api_base = os.getenv('DEFAULT_API_BASE', '')
        api_key = os.getenv('DEFAULT_API_KEY', '')
        
        # Prepare all candidate data for evaluation
        # First, collect all functional_core_rmsd values for normalization
        all_rmsd_values = []
        for row in enriched_rows:
            rmsd_val = row.get('functional_core_rmsd')
            if rmsd_val and rmsd_val != '':
                try:
                    all_rmsd_values.append(float(rmsd_val))
                except (ValueError, TypeError):
                    pass
        
        # Calculate normalization parameters for RMSD (assume max reasonable RMSD is 20 Å)
        max_rmsd = max(all_rmsd_values) if all_rmsd_values else 20.0
        max_rmsd = max(max_rmsd, 20.0)  # Ensure at least 20 Å as max
        min_rmsd = min(all_rmsd_values) if all_rmsd_values else 0.0
        
        all_candidate_data = []
        for i, row in enumerate(enriched_rows):
            # Normalize functional_core_rmsd (invert so higher is better, then normalize to 0-1)
            functional_core_rmsd_raw = None
            functional_core_rmsd_norm = None
            if row.get('functional_core_rmsd') and row.get('functional_core_rmsd') != '':
                try:
                    functional_core_rmsd_raw = float(row.get('functional_core_rmsd'))
                    # Invert: smaller RMSD -> larger normalized value
                    # Normalize: (max - value) / (max - min), then scale to 0-1
                    if max_rmsd > min_rmsd:
                        functional_core_rmsd_norm = (max_rmsd - functional_core_rmsd_raw) / (max_rmsd - min_rmsd)
                        functional_core_rmsd_norm = max(0.0, min(1.0, functional_core_rmsd_norm))  # Clamp to [0, 1]
                    else:
                        functional_core_rmsd_norm = 1.0 if functional_core_rmsd_raw < 2.0 else 0.0
                except (ValueError, TypeError):
                    pass
            
            candidate = {
                "index": i + 1,
                "pred_name": row.get('pred_name', ''),
                "alignment_description": row.get('alignment_description', ''),
                "pseudoknot_alignment_description": row.get('pseudoknot_alignment_description', ''),
                "motif_ss_alignment_description": row.get('motif_ss_alignment_description', ''),
                "tm_score": float(row.get('tm_score', 0)) if row.get('tm_score') else None,
                "ss_f1score": float(row.get('ss_f1score', 0)) if row.get('ss_f1score') else None,
                "functional_core_f1score": float(row.get('functional_core_f1score', 0)) if row.get('functional_core_f1score') else None,
                "pseudoknot_recall": float(row.get('pseudoknot_recall', 0)) if row.get('pseudoknot_recall') else None,
                "tmscore_wt": float(row.get('tmscore_wt', 0)) if row.get('tmscore_wt') else None,
                "functional_core_rmsd": functional_core_rmsd_raw,  # Keep raw value for display
                "functional_core_rmsd_norm": functional_core_rmsd_norm,  # Normalized value for scoring
                "seq_rec_rate": float(row.get('seq_rec_rate', 0)) if row.get('seq_rec_rate') else None,
                "seq_similarity": float(row.get('seq_similarity', 0)) if row.get('seq_similarity') else None,
                "ss_pairs_ratio": float(row.get('ss_pairs_ratio', 0)) if row.get('ss_pairs_ratio') else None,
                "pseudoknot_ss_pairs_ratio": float(row.get('pseudoknot_ss_pairs_ratio', 0)) if row.get('pseudoknot_ss_pairs_ratio') else None,
            }
            all_candidate_data.append(candidate)
        
        # Step 1: Parallel LLM evaluation for each candidate (max 20 concurrent)
        print(f"[INFO] Starting parallel LLM evaluation for {len(all_candidate_data)} candidates...", file=sys.stderr)
        evaluation_results = {}
        max_workers = min(20, len(all_candidate_data))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_single_candidate,
                    candidate,
                    candidate['index'],
                    model_name,
                    api_base,
                    api_key,
                ): candidate['index']
                for candidate in all_candidate_data
            }
            
            for future in as_completed(futures):
                try:
                    idx, eval_result = future.result()
                    evaluation_results[idx] = eval_result
                    print(f"[INFO] Completed evaluation for candidate #{idx}: score={eval_result['score']}", file=sys.stderr)
                except Exception as e:
                    idx = futures[future]
                    print(f"[ERROR] Failed to evaluate candidate #{idx}: {str(e)}", file=sys.stderr)
                    evaluation_results[idx] = {'score': 5, 'reason': f"Evaluation error: {str(e)[:100]}"}
        
        # Store evaluation results in enriched_rows
        for i, row in enumerate(enriched_rows, 1):
            if i in evaluation_results:
                row['llm_score'] = evaluation_results[i]['score']
                row['llm_evaluation_reason'] = evaluation_results[i]['reason']
            else:
                row['llm_score'] = 5
                row['llm_evaluation_reason'] = "Evaluation not completed"
        
        # Step 2: Prepare evaluation results for weight determination
        evaluation_data = []
        for candidate in all_candidate_data:
            idx = candidate['index']
            eval_result = evaluation_results.get(idx, {'score': 5, 'reason': 'Not evaluated'})
            evaluation_data.append({
                "index": idx,
                "pred_name": candidate['pred_name'],
                "llm_score": eval_result['score'],
                "llm_evaluation_reason": eval_result['reason'],
                "tm_score": candidate['tm_score'],
                "ss_f1score": candidate['ss_f1score'],
                "functional_core_f1score": candidate['functional_core_f1score'],
                "pseudoknot_recall": candidate['pseudoknot_recall'],
                "tmscore_wt": candidate['tmscore_wt'],
                "functional_core_rmsd": candidate['functional_core_rmsd'],
                "functional_core_rmsd_norm": candidate['functional_core_rmsd_norm'],
                "seq_rec_rate": candidate['seq_rec_rate'],
                "seq_similarity": candidate['seq_similarity'],
                "ss_pairs_ratio": candidate['ss_pairs_ratio'],
                "pseudoknot_ss_pairs_ratio": candidate['pseudoknot_ss_pairs_ratio'],
            })
        
        # Step 2: Use LLM to determine weights based on evaluation results
        system_prompt = (
            "You are an RNA structure evaluation expert. You need to determine weights for a filtering task.\n"
            "Task: Based on LLM evaluation results for each candidate, determine optimal weights for different metrics.\n\n"
            "You have received LLM evaluations (scores 0-10 and reasons) for each candidate, along with their metric values.\n"
            "Your task is to analyze which metrics are most correlated with high LLM scores and determine appropriate weights.\n\n"
            "Key Metrics (PRIORITY ORDER - these should receive higher weights if they correlate with high LLM scores):\n"
            "1. tm_score: Higher is better (structural similarity, range 0-1) - CRITICAL\n"
            "2. ss_f1score: Higher is better (secondary structure F1-score, range 0-1) - CRITICAL\n"
            "3. functional_core_f1score: Higher is better (functional core region F1-score, range 0-1) - CRITICAL\n"
            "4. pseudoknot_recall: Higher is better (pseudoknot recall, range 0-1) - CRITICAL\n"
            "5. tmscore_wt: Higher is better (TM-score compared to wild-type structure, range 0-1) - CRITICAL\n"
            "6. functional_core_rmsd_norm: Higher is better (normalized functional core RMSD after Kabsch alignment, range 0-1) - CRITICAL. "
            "This is the normalized and inverted version of the functional core RMSD. "
            "The original RMSD (in Angstroms) measures structural deviation - smaller RMSD values indicate better structural accuracy. "
            "After normalization and inversion, values closer to 1.0 indicate excellent structural accuracy (< 2.0 Å), "
            "values around 0.7-0.9 indicate good accuracy (2.0-5.0 Å), values around 0.4-0.7 indicate moderate accuracy (5.0-10.0 Å), "
            "and values < 0.4 indicate poor accuracy (> 10.0 Å).\n\n"
            "Secondary Metrics:\n"
            "7. seq_rec_rate: Lower is better (sequence recovery rate, range 0-1, lower indicates greater sequence difference, potentially more innovative)\n"
            "8. seq_similarity: Lower is better (sequence similarity, range 0-1)\n"
            "9. ss_pairs_ratio: Higher is better (ratio of canonical pairs in secondary structure, range 0-1)\n"
            "10. pseudoknot_ss_pairs_ratio: Higher is better (ratio of canonical pairs in pseudoknot regions, range 0-1)\n\n"
            "For each candidate, you will receive:\n"
            "- index: Candidate index\n"
            "- pred_name: Prediction name\n"
            "- llm_score: LLM evaluation score (0-10)\n"
            "- llm_evaluation_reason: LLM's detailed evaluation reason\n"
            "- All numeric metrics listed above\n\n"
            "Please analyze the relationship between LLM scores and metric values to determine optimal weights.\n"
            "Metrics that strongly correlate with high LLM scores should receive higher weights.\n\n"
            "Return JSON format with all weights (weights should sum to 1.0):\n"
            "{\n"
            '  "weight_tm_score": float (0.1-0.15),\n'
            '  "weight_ss_f1score": float (0.3-0.6),\n'
            '  "weight_functional_core_f1score": float (0.3-0.6),\n'
            '  "weight_pseudoknot_recall": float (0.3-0.6),\n'
            '  "weight_tmscore_wt": float (0.3-0.6),\n'
            '  "weight_functional_core_rmsd_norm": float (0.3-0.6),\n'
            '  "weight_seq_rec_rate": float (0.05-0.1),\n'
            '  "weight_seq_similarity": float (0.0-0.05),\n'
            '  "weight_ss_pairs_ratio": float (0.0-0.05),\n'
            '  "weight_pseudoknot_ss_pairs_ratio": float (0.0-0.05),\n'
            '  "reason": string\n'
            "}\n"
            "Note: The weights should be different from each other to emphasize the importance of different metrics. Do not set the weights to the same value."
            "The reason field should explain the rationale for weight selection based on the correlation analysis."
        )
        
        user_prompt = (
            f"Total candidates evaluated: {len(evaluation_data)}\n\n"
            f"LLM Evaluation Statistics:\n"
        )
        
        llm_scores = [e['llm_score'] for e in evaluation_data]
        if llm_scores:
            user_prompt += (
                f"- LLM score range: {min(llm_scores)} - {max(llm_scores)}, "
                f"average: {sum(llm_scores) / len(llm_scores):.2f}\n"
            )
        
        user_prompt += "\nAll candidate evaluation results (JSON format):\n"
        evaluation_json = json.dumps(evaluation_data, ensure_ascii=False, indent=2)
        user_prompt += evaluation_json
        
        user_prompt += "\n\nPlease analyze the correlation between LLM scores and metric values to determine optimal weights for all metrics."
        
        try:
            # 构建 LLM 配置，确保使用 OAI 类型（OpenAI 兼容 API）
            llm_cfg = {'model': model_name}
            if api_base:
                llm_cfg['model_server'] = api_base
                # 明确指定使用 OAI 类型，避免模型名包含 "qwen" 时被识别为 dashscope
                llm_cfg['model_type'] = 'oai'
            else:
                # 如果没有 api_base，也设置 model_type 为 oai（如果模型名包含 qwen，会被识别为 dashscope）
                llm_cfg['model_type'] = 'oai'
            if api_key:
                llm_cfg['api_key'] = api_key
            
            llm = get_chat_model(llm_cfg)
            *_, responses = llm.chat(messages=[
                Message(role=SYSTEM, content=system_prompt),
                Message(role=USER, content=user_prompt),
            ])
            content = responses[-1].content if responses else ""
            
            # Extract JSON - try multiple methods
            json_content = content.strip()
            
            # Method 1: Try to extract JSON from markdown code blocks
            if '```json' in json_content:
                json_content = json_content.split('```json')[1].split('```')[0].strip()
            elif '```' in json_content:
                # Might be a code block starting with ```
                parts = json_content.split('```')
                if len(parts) >= 3:
                    json_content = parts[1].strip()
                    # If first part is language identifier, take second part
                    if json_content.startswith('json'):
                        json_content = json_content[4:].strip()
                    elif not json_content.startswith('{'):
                        # If first part is not JSON, try second part
                        if len(parts) >= 3 and parts[2].strip().startswith('{'):
                            json_content = parts[2].strip()
            
            # Method 2: Try to directly find JSON object
            if not json_content.startswith('{'):
                # Find first { and last }
                start_idx = json_content.find('{')
                end_idx = json_content.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_content = json_content[start_idx:end_idx+1]
            
            # If still cannot find JSON, try using regex
            if not json_content or not json_content.startswith('{'):
                import re
                json_match = re.search(r'\{[^{}]*"weight_tm_score"[^{}]*\}', json_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
            
            # Parse JSON
            if not json_content or not json_content.strip():
                raise ValueError(f"Unable to extract JSON from LLM response. Original content: {content[:200]}")
            
            weight_result = json.loads(json_content)
            
            # Extract all weights with defaults
            weight_tm = float(weight_result.get('weight_tm_score', 0.15))
            weight_ss_f1 = float(weight_result.get('weight_ss_f1score', 0.15))
            weight_func_core = float(weight_result.get('weight_functional_core_f1score', 0.15))
            weight_pk_recall = float(weight_result.get('weight_pseudoknot_recall', 0.10))
            weight_tm_wt = float(weight_result.get('weight_tmscore_wt', 0.15))
            weight_func_rmsd_norm = float(weight_result.get('weight_functional_core_rmsd_norm', 0.15))
            weight_seq_rec = float(weight_result.get('weight_seq_rec_rate', 0.08))
            weight_seq_sim = float(weight_result.get('weight_seq_similarity', 0.02))
            weight_ss_pairs = float(weight_result.get('weight_ss_pairs_ratio', 0.03))
            weight_pk_pairs = float(weight_result.get('weight_pseudoknot_ss_pairs_ratio', 0.02))
            weight_reason = weight_result.get('reason', '')
            
            # Normalize all weights to sum to 1.0
            all_weights = [
                weight_tm, weight_ss_f1, weight_func_core, weight_pk_recall,
                weight_tm_wt, weight_func_rmsd_norm,
                weight_seq_rec, weight_seq_sim, weight_ss_pairs, weight_pk_pairs
            ]
            total_weight = sum(all_weights)
            if total_weight > 0:
                weight_tm /= total_weight
                weight_ss_f1 /= total_weight
                weight_func_core /= total_weight
                weight_pk_recall /= total_weight
                weight_tm_wt /= total_weight
                weight_func_rmsd_norm /= total_weight
                weight_seq_rec /= total_weight
                weight_seq_sim /= total_weight
                weight_ss_pairs /= total_weight
                weight_pk_pairs /= total_weight
            else:
                # Fallback to default weights
                weight_tm, weight_ss_f1, weight_func_core, weight_pk_recall = 0.15, 0.15, 0.15, 0.10
                weight_tm_wt, weight_func_rmsd_norm = 0.15, 0.15
                weight_seq_rec, weight_seq_sim, weight_ss_pairs, weight_pk_pairs = 0.08, 0.02, 0.03, 0.02
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"[ERROR] LLM weight determination failed: {error_msg}", file=sys.stderr)
            print(f"[ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
            return json.dumps({
                "success": False,
                "error": f"LLM weight determination failed: {error_msg}",
                "input_csv": str(input_path),
            }, ensure_ascii=False)
    
    # Calculate composite scores and sort
    # Higher is better: tm_score, ss_f1score, functional_core_f1score, pseudoknot_recall, tmscore_wt, functional_core_rmsd_norm, ss_pairs_ratio, pseudoknot_ss_pairs_ratio
    # Lower is better: seq_rec_rate, seq_similarity
    scored_rows = []
    
    # Recalculate normalization for composite scoring (in case weights were provided externally)
    all_rmsd_values = []
    for row in enriched_rows:
        rmsd_val = row.get('functional_core_rmsd')
        if rmsd_val and rmsd_val != '':
            try:
                all_rmsd_values.append(float(rmsd_val))
            except (ValueError, TypeError):
                pass
    max_rmsd = max(all_rmsd_values) if all_rmsd_values else 20.0
    max_rmsd = max(max_rmsd, 20.0)
    min_rmsd = min(all_rmsd_values) if all_rmsd_values else 0.0
    
    for row in enriched_rows:
        try:
            tm_score = float(row.get('tm_score', 0)) if row.get('tm_score') else 0
            ss_f1score = float(row.get('ss_f1score', 0)) if row.get('ss_f1score') else 0
            func_core_f1score = float(row.get('functional_core_f1score', 0)) if row.get('functional_core_f1score') else 0
            pk_recall = float(row.get('pseudoknot_recall', 0)) if row.get('pseudoknot_recall') else 0
            tm_wt = float(row.get('tmscore_wt', 0)) if row.get('tmscore_wt') else 0
            
            # Normalize functional_core_rmsd if not already normalized
            func_rmsd_norm = 0.0
            if row.get('functional_core_rmsd') and row.get('functional_core_rmsd') != '':
                try:
                    func_rmsd_raw = float(row.get('functional_core_rmsd'))
                    if max_rmsd > min_rmsd:
                        func_rmsd_norm = (max_rmsd - func_rmsd_raw) / (max_rmsd - min_rmsd)
                        func_rmsd_norm = max(0.0, min(1.0, func_rmsd_norm))
                    else:
                        func_rmsd_norm = 1.0 if func_rmsd_raw < 2.0 else 0.0
                except (ValueError, TypeError):
                    pass
            
            seq_rec_rate = float(row.get('seq_rec_rate', 0)) if row.get('seq_rec_rate') else 0
            seq_similarity = float(row.get('seq_similarity', 0)) if row.get('seq_similarity') else 0
            ss_pairs_ratio = float(row.get('ss_pairs_ratio', 0)) if row.get('ss_pairs_ratio') else 0
            pk_pairs_ratio = float(row.get('pseudoknot_ss_pairs_ratio', 0)) if row.get('pseudoknot_ss_pairs_ratio') else 0
            
            composite_score = (
                weight_tm * tm_score +
                weight_ss_f1 * ss_f1score +
                weight_func_core * func_core_f1score +
                weight_pk_recall * pk_recall +
                weight_tm_wt * tm_wt +
                weight_func_rmsd_norm * func_rmsd_norm +
                weight_ss_pairs * ss_pairs_ratio +
                weight_pk_pairs * pk_pairs_ratio -
                weight_seq_rec * seq_rec_rate -
                weight_seq_sim * seq_similarity
            )
            scored_rows.append((composite_score, row))
        except (ValueError, KeyError):
            continue
    
    # Sort by composite score in descending order
    scored_rows.sort(key=lambda x: x[0], reverse=True)
    
    # Take top-k
    top_k = max(1, min(top_k, len(scored_rows)))
    top_rows = [row for _, row in scored_rows[:top_k]]
    
    # Determine output path
    if output_csv is None:
        input_dir = input_path.parent
        output_path = input_dir / f"{input_path.stem}_llm_filtered.csv"
    else:
        output_path = Path(output_csv)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write filtered results (preserve all original CSV fields)
    if top_rows:
        fieldnames = list(top_rows[0].keys())
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_rows)
    
    return json.dumps({
        "success": True,
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "total_count": len(enriched_rows),
        "filtered_count": len(top_rows),
        "weights": {
            "weight_tm_score": weight_tm,
            "weight_ss_f1score": weight_ss_f1,
            "weight_functional_core_f1score": weight_func_core,
            "weight_pseudoknot_recall": weight_pk_recall,
            "weight_tmscore_wt": weight_tm_wt,
            "weight_functional_core_rmsd_norm": weight_func_rmsd_norm,
            "weight_seq_rec_rate": weight_seq_rec,
            "weight_seq_similarity": weight_seq_sim,
            "weight_ss_pairs_ratio": weight_ss_pairs,
            "weight_pseudoknot_ss_pairs_ratio": weight_pk_pairs,
            "reason": weight_reason,
        },
        "top_k": top_k,
        "best_result": {
            "pred_name": top_rows[0].get('pred_name', '') if top_rows else None,
            "tm_score": float(top_rows[0].get('tm_score', 0)) if top_rows and top_rows[0].get('tm_score') else None,
            "ss_f1score": float(top_rows[0].get('ss_f1score', 0)) if top_rows and top_rows[0].get('ss_f1score') else None,
            "functional_core_f1score": float(top_rows[0].get('functional_core_f1score', 0)) if top_rows and top_rows[0].get('functional_core_f1score') else None,
            "pseudoknot_recall": float(top_rows[0].get('pseudoknot_recall', 0)) if top_rows and top_rows[0].get('pseudoknot_recall') else None,
            "seq_rec_rate": float(top_rows[0].get('seq_rec_rate', 0)) if top_rows and top_rows[0].get('seq_rec_rate') else None,
            "llm_score": int(top_rows[0].get('llm_score', 0)) if top_rows and top_rows[0].get('llm_score') else None,
            "composite_score": scored_rows[0][0] if scored_rows else None,
        } if top_rows else None,
    }, ensure_ascii=False)


@register_tool('llm_weighted_filter_v2')
class LLMWeightedFilterTool(BaseTool):
    """Use LLM to determine weights and further filter top-k records from first-round filtered CSV"""

    description = (
        "From first-round filtered CSV file, use two-stage LLM evaluation to filter top-k records (default 20). "
        "Stage 1: Parallel LLM evaluation (max 20 concurrent) - Each candidate is evaluated individually by LLM "
        "based on metrics and natural language descriptions, receiving a score (0-10) and evaluation reason. "
        "Stage 2: Weight determination - All evaluation results are analyzed by LLM to determine optimal weights "
        "for different metrics based on correlation with LLM scores. "
        "Stage 3: Composite scoring - Calculate weighted composite scores and select top-k. "
        "Enhanced version includes: (1) Alignment descriptions from USalign, (2) Pseudoknot alignment analysis, "
        "(3) Motif region secondary structure analysis. "
        "Key metrics (high priority): tm_score, ss_f1score, functional_core_f1score, pseudoknot_recall. "
        "Secondary metrics: seq_rec_rate (lower better), seq_similarity (lower better), ss_pairs_ratio, pseudoknot_ss_pairs_ratio. "
        "If weight_tm_score and weight_seq_rec_rate parameters are provided, skip LLM evaluation and use external weights directly (backward compatibility mode)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "input_csv": {
                "type": "string",
                "description": "Input filtered CSV file path (from first_round_filter_tool output)",
            },
            "output_csv": {
                "type": "string",
                "description": "Output final filtered CSV file path (optional, default generates *_llm_filtered.csv in same directory as input file)",
            },
            "top_k": {
                "type": "integer",
                "default": 20,
                "description": "Return top-k records (default 20)",
            },
            "model": {
                "type": "string",
                "description": "LLM model name (optional, default read from environment variables)",
            },
            "weight_tm_score": {
                "type": "number",
                "description": "Externally provided TM-score weight (optional, if provided skips LLM weight determination, usually from wet_lab_analysis_tool)",
            },
            "weight_seq_rec_rate": {
                "type": "number",
                "description": "Externally provided seq_rec_rate weight (optional, if provided skips LLM weight determination, usually from wet_lab_analysis_tool)",
            },
            "weight_reason": {
                "type": "string",
                "description": "Reason for weight selection (optional, used with external weights)",
            },
            "usalign_bin": {
                "type": "string",
                "description": "Path to USalign binary (optional, defaults to DEFAULT_USALIGN_BIN)",
            },
        },
        "required": ["input_csv"],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return llm_weighted_filter(
            input_csv=args["input_csv"],
            output_csv=args.get("output_csv"),
            top_k=args.get("top_k", 20),
            model=args.get("model"),
            weight_tm_score=args.get("weight_tm_score"),
            weight_seq_rec_rate=args.get("weight_seq_rec_rate"),
            weight_reason=args.get("weight_reason"),
            usalign_bin=args.get("usalign_bin", DEFAULT_USALIGN_BIN),
        )

