"""
RNApolis tool module
For extracting RNA secondary structure from PDB/CIF files and converting to Dot-Bracket Notation
"""

import json
from rhoagent.tools.base import BaseTool, register_tool

from .utils.rnapolis_utils import (
    batch_pdb_to_dotbracket,
)


@register_tool('batch_pdb_to_dotbracket')
class BatchPdbToDotBracketTool(BaseTool):
    description = 'Secondary structure extraction tool, batch convert PDB/CIF files in directory to dot-bracket notation.'
    parameters = {
        'type': 'object',
        'properties': {
            'input_dir': {
                'type': 'string',
                'description': 'Input directory'
            },
            'output_dir': {
                'type': 'string',
                'description': 'Output directory'
            },
            'file_pattern': {
                'type': 'string',
                'default': '*.pdb',
                'description': 'File matching pattern'
            },
        },
        'required': ['input_dir', 'output_dir'],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return batch_pdb_to_dotbracket(
            input_dir=args['input_dir'],
            output_dir=args['output_dir'],
            file_pattern=args.get('file_pattern', '*.pdb'),
        )



