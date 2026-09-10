"""
Structure evaluation tool module
For evaluating similarity between predicted and ground truth structures, integrated with RhoAgent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rhoagent.tools.base import BaseTool, register_tool

from .utils.eval_utils import evaluate_batch, DEFAULT_USALIGN_BIN


@register_tool('eval_rna_struct_batch')
class EvalRNAStructBatchTool(BaseTool):
    """Batch structure evaluation tool for RNA structures"""

    description = 'Batch compare all predicted structures with ground truth structures using USalign, outputting TM-score, RMSD, sequence recovery rate, and sequence similarity for each structure. If target_ss is provided, also calculates secondary structure F1-score, AU/GC/GU pair ratios in target structure, and AU/GC/GU pair ratios in pseudoknot regions. If fixed_motif is provided (e.g., "------AAU----UUUA-----"), calculates functional_core_f1score for the fixed motif region. Results are saved to a CSV file (results.csv) in the output directory.'
    parameters = {
        'type': 'object',
        'properties': {
            'gt_struc_dir': {
                'type': 'string',
                'description': 'Ground truth structure file directory path'
            },
            'gt_seq': {
                'type': 'string',
                'description': 'Ground truth sequence'
            },
            'pred_struc_dir': {
                'type': 'string',
                'description': 'Directory path containing predicted structure files (recursively searches for all .pdb/.cif files)'
            },
            'pred_seq_dir': {
                'type': 'string',
                'description': 'Directory path containing predicted sequence FASTA files (optional, for sequence recovery rate calculation)'
            },
            'target_ss': {
                'type': 'string',
                'description': 'Target secondary structure (dot-bracket notation, e.g., "(((...)))" or with pseudoknots "(((...[[[...]]]...)))")'
            },
            'fixed_motif': {
                'type': 'string',
                'description': 'Fixed motif region (e.g., "------AAU----UUUA-----", non-"-" characters indicate fixed_motif positions)'
            },
            'output_dir': {
                'type': 'string',
                'description': 'Output results directory path'
            },
        },
        'required': ['gt_struc_dir', 'gt_seq', 'pred_struc_dir', 'pred_seq_dir'],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        # evaluate_batch 会调用 evaluate_structure 和 evaluate_sequence
        result = evaluate_batch(
            gt_struc_dir=args['gt_struc_dir'],
            pred_struc_dir=args['pred_struc_dir'],
            pred_seq_dir=args.get('pred_seq_dir'),
            output_dir=args.get('output_dir'),
            gt_seq=args.get('gt_seq'),
            target_ss=args.get('target_ss'),
            fixed_motif=args.get('fixed_motif'),
        )
        return json.dumps(result, ensure_ascii=False)
