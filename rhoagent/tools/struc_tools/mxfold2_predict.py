"""
mxfold2 tool module
For predicting RNA secondary structure, compatible with RhoAgent
"""

from __future__ import annotations

import json
from typing import Optional

from rhoagent.tools.base import BaseTool, register_tool

from .utils.mxfold2_utils import predict_secondary_structure


def mxfold2_predict(input_dir: str,
                    output_dir: str,
                    mxfold2_bin: str = 'mxfold2') -> str:
    """Use mxfold2 to predict RNA secondary structure. Input directory containing FASTA files, save results to specified output directory."""
    import glob
    import os

    try:
        # Create output directory
        pred_dir = os.path.join(output_dir, 'mxfold_pred')
        os.makedirs(pred_dir, exist_ok=True)

        # Find all FASTA files
        fasta_files = glob.glob(os.path.join(input_dir, '*.fasta')) + \
                      glob.glob(os.path.join(input_dir, '*.fa')) + \
                      glob.glob(os.path.join(input_dir, '*.fas'))
        
        if not fasta_files:
            return json.dumps({
                'error': f"No FASTA files found in directory {input_dir}",
                'success': False,
                'input_dir': input_dir
            }, ensure_ascii=False, indent=2)

        results = []
        success_count = 0
        fail_count = 0

        for fasta_file in fasta_files:
            try:
                validation = validate_fasta_file(fasta_file)
                if not validation['valid']:
                    results.append({
                        'file': fasta_file,
                        'status': 'failed',
                        'error': f"Validation failed: {validation['error']}"
                    })
                    fail_count += 1
                    continue

                # Use filename as subdirectory or filename prefix
                file_name = os.path.splitext(os.path.basename(fasta_file))[0]
                file_output_dir = os.path.join(pred_dir, file_name)
                os.makedirs(file_output_dir, exist_ok=True)

                # Call prediction function (assuming predict_secondary_structure can handle single file and output)
                # Note: Need to ensure mxfold2_utils.predict_secondary_structure supports output_dir parameter
                # If not supported, may need to adjust the calling method here
                pred_result = predict_secondary_structure(
                    fasta_file=fasta_file,
                    output_dir=file_output_dir,
                    mxfold2_bin=mxfold2_bin,
                )
                
                results.append({
                    'file': fasta_file,
                    'status': 'success',
                    'output_path': file_output_dir,
                    'prediction': pred_result
                })
                success_count += 1

            except Exception as e:
                results.append({
                    'file': fasta_file,
                    'status': 'failed',
                    'error': str(e)
                })
                fail_count += 1

        summary = {
            'success': True,
            'input_dir': input_dir,
            'output_dir': pred_dir,
            'total_files': len(fasta_files),
            'success_count': success_count,
            'fail_count': fail_count,
            'details': results
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            'error': str(e),
            'input_dir': input_dir,
            'output_dir': output_dir,
            'mxfold2_bin': mxfold2_bin
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)



@register_tool('mxfold2_predict')
class Mxfold2PredictTool(BaseTool):
    """Tool for batch secondary structure prediction using MXfold2"""

    description = 'Batch call mxfold2 to predict RNA secondary structure for all FASTA files in directory, output dot-bracket structure and scoring information to specified directory.'
    parameters = {
        'type': 'object',
        'properties': {
            'input_dir': {
                'type': 'string',
                'description': 'Input directory containing FASTA files'
            },
            'output_dir': {
                'type': 'string',
                'description': 'Output directory, results will be saved to {output_dir}/mxfold_pred/'
            },
            'mxfold2_bin': {
                'type': 'string',
                'default': 'mxfold2',
                'description': 'Path to mxfold2 executable'
            },
        },
        'required': ['input_dir', 'output_dir'],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return mxfold2_predict(
            input_dir=args['input_dir'],
            output_dir=args['output_dir'],
            mxfold2_bin=args.get('mxfold2_bin', 'mxfold2'),
        )

