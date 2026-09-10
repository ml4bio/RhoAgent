"""
RNApolis tool module
For extracting RNA secondary structure from PDB/CIF files and converting to Dot-Bracket Notation
"""

import json
from rhoagent.tools.base import BaseTool, register_tool

from .utils.rnapolis_utils import (
    analyze_rna_structure,
)

@register_tool('analyze_rna_structure')
class AnalyzeRNAStructureTool(BaseTool):
    description = 'Read and analyze RNA structure, return chain, residue, and base pair statistics.'
    parameters = {
        'type': 'object',
        'properties': {
            'pdb_path': {
                'type': 'string',
                'description': 'Input structure file path'
            },
            'detailed_analysis': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to perform detailed analysis'
            },
        },
        'required': ['pdb_path'],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return analyze_rna_structure(
            pdb_path=args['pdb_path'],
            detailed_analysis=args.get('detailed_analysis', True),
        )