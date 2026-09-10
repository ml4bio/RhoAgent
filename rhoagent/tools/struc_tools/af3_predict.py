"""
AF3 preparation tool: generate AlphaFold3 JSON inputs and a run script based on a template, and execute the script to predict the structure.
"""

from __future__ import annotations

import os
import json
import glob
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from Bio import SeqIO

from rhoagent.tools.base import BaseTool, register_tool

_AF3_JSON_TEMPLATE = {
    "dialect": "alphafold3",
    "version": 2,
    "name": "{name}",
    "sequences": "__SEQUENCES__",
    "modelSeeds": "__SEEDS__",
    "bondedAtomPairs": None,
    "userCCD": None,
}


def _build_sequences_block(rna_id: str, sequence: str, ligands: List[Dict]) -> List[Dict]:
    seqs: List[Dict] = [
        {
            "rna": {
                "id": rna_id,
                "sequence": sequence,
                "modifications": [],
                "unpairedMsa": sequence,
            }
        }
    ]
    for lig in ligands:
        seqs.append({
            "ligand": {
                "id": lig.get("id", "L1"),
                "ccdCodes": lig.get("ccdCodes", []),
            }
        })
    return seqs


def af3_prediction_pipeline(
    input_dir: str,
    json_dir: str,
    output_dir: str,
    rna_id: str = "A",
    ligands: List[Dict] = None,
    model_seeds: Optional[List[int]] = None,
    gpu_id: int = 0,
    script_path: Optional[str] = None,
) -> str:
    """Batch generate AF3 JSON files from a directory of FASTA files."""
    if ligands is None:
        ligands = []
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all FASTA files
    fasta_files = (
        glob.glob(os.path.join(input_dir, "*.fasta"))
        + glob.glob(os.path.join(input_dir, "*.fas"))
        + glob.glob(os.path.join(input_dir, "*.fa"))
    )
    fasta_files = sorted(set(fasta_files))
    
    if not fasta_files:
        return json.dumps({
            "success": False,
            "error": f"No FASTA files found in {input_dir}",
            "json_paths": [],
            "script_path": None,
            "output_dir": output_dir,
        }, ensure_ascii=False)
    
    seeds = model_seeds if model_seeds else [1]
    json_paths = []
    
    # Process each FASTA file
    for fasta_file in fasta_files:
        seq_name = Path(fasta_file).stem
        try:
            # Read sequence from FASTA file
            with open(fasta_file, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    sequence_str = str(record.seq)
                    
                    # Generate JSON for this sequence
                    json_name = f"{seq_name}.json"
                    json_path = os.path.join(json_dir, json_name)
                    
                    content = json.loads(json.dumps(_AF3_JSON_TEMPLATE))
                    content["name"] = seq_name
                    content["sequences"] = _build_sequences_block(rna_id, sequence_str, ligands)
                    content["modelSeeds"] = seeds
                    
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(content, f, ensure_ascii=False, indent=2)
                    
                    json_paths.append(json_path)
                    break  # Only process first sequence in each FASTA file
        except Exception as e:
            print(f"Warning: Failed to process {fasta_file}: {e}")
            continue
    
    if not json_paths:
        return json.dumps({
            "success": False,
            "error": "Failed to generate any JSON files",
            "json_paths": [],
            "script_path": None,
            "output_dir": output_dir,
        }, ensure_ascii=False)
    
    # Generate unified script
    script_target = script_path or os.getenv(
        'AF3_SCRIPT_PATH', os.path.join(os.path.dirname(json_dir), 'run_af3.sh'))
    
    # Use absolute paths
    json_dir_abs = os.path.abspath(json_dir)
    output_dir_abs = os.path.abspath(output_dir)
    
    script = f"""#!/bin/bash

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95

INPUT_DIR="{json_dir_abs}"
OUTPUT_DIR="{output_dir_abs}"
GPU_ID={gpu_id}

echo "AF3: Starting predictions..."
echo "Input directory: ${{INPUT_DIR}}"
echo "Output directory: ${{OUTPUT_DIR}}"
echo "Total JSON files: {len(json_paths)}"

CUDA_VISIBLE_DEVICES=${{GPU_ID}} /workspace/shared/envs/rna/af3/bin/python3 /workspace/shared/code/rna/alphafold3/run_alphafold.py \\
  --model_dir=/workspace/shared/code/rna/alphafold3/alphafold_param/ \\
  --db_dir=/workspace/shared/code/rna/alphafold3/alphafold_db/ \\
  --jax_compilation_cache_dir=/workspace/shared/code/rna/alphafold3/alphafold_cache/ \\
  --nhmmer_binary_path=/workspace/shared/code/rna/alphafold3/alphafold_bin/hmmer-3.4/bin/nhmmer \\
  --hmmalign_binary_path=/workspace/shared/code/rna/alphafold3/alphafold_bin/hmmer-3.4/bin/hmmalign \\
  --hmmbuild_binary_path=/workspace/shared/code/rna/alphafold3/alphafold_bin/hmmer-3.4/bin/hmmbuild \\
  --hmmsearch_binary_path=/workspace/shared/code/rna/alphafold3/alphafold_bin/hmmer-3.4/bin/hmmsearch \\
  --jackhmmer_binary_path=/workspace/shared/code/rna/alphafold3/alphafold_bin/hmmer-3.4/bin/jackhmmer \\
  --input_dir="${{INPUT_DIR}}" \\
  --output_dir="${{OUTPUT_DIR}}" \\
  2>&1 | tee "${{OUTPUT_DIR}}/af3_generated.log"

if [ $? -eq 0 ]; then
  echo "AF3: All predictions completed successfully."
else
  echo "AF3: ERROR occurred during prediction."
  exit 1
fi
"""
    
    with open(script_target, "w", encoding="utf-8") as f:
        f.write(script)
    os.chmod(script_target, 0o755)
    
    # Execute the script silently
    try:
        result = subprocess.run(
            ["bash", script_target],
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
        )
        script_success = result.returncode == 0
        script_error = result.stderr if not script_success else None
    except Exception as e:
        script_success = False
        script_error = str(e)
    
    return json.dumps({
        "success": True,
        "input_dir": input_dir,
        "json_dir": json_dir,
        "json_paths": json_paths,
        "num_sequences": len(json_paths),
        "script_path": script_target,
        "output_dir": output_dir,
        "script_executed": True,
        "script_success": script_success,
        "script_error": script_error,
    }, ensure_ascii=False)


@register_tool("af3_prediction_pipeline")
class AF3PreidctionPipelineTool(BaseTool):
    """Tool for 3D structure prediction using AlphaFold3"""

    description = '3D structure prediction tool, batch generate AlphaFold3 JSON input files and run script from directory containing FASTA files, and automatically execute the script for structure prediction.'
    parameters = {
        'type': 'object',
        'properties': {
            'input_dir': {
                'type': 'string',
                'description': 'Input directory containing FASTA files'
            },
            'json_dir': {
                'type': 'string',
                'description': 'JSON output directory'
            },
            'output_dir': {
                'type': 'string',
                'description': 'AF3 execution output directory'
            },
            'rna_id': {
                'type': 'string',
                'default': 'A',
                'description': 'RNA chain ID, e.g., "A"'
            },
            'ligands': {
                'type': 'array',
                'description': 'Ligand list, each element contains id and ccdCodes fields',
            },
            'model_seeds': {
                'type': 'array',
                'description': 'Model random seed list',
            },
            'gpu_id': {
                'type': 'integer',
                'default': 0,
                'description': 'GPU ID to use in run script'
            },
            'script_path': {
                'type': 'string',
                'description': 'Optional: Script output path, defaults to AF3_SCRIPT_PATH or preset location'
            },
        },
        'required': ['input_dir', 'json_dir', 'output_dir'],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return af3_prediction_pipeline(
            input_dir=args['input_dir'],
            json_dir=args['json_dir'],
            output_dir=args['output_dir'],
            rna_id=args.get('rna_id', 'A'),
            ligands=args.get('ligands', []),
            model_seeds=args.get('model_seeds'),
            gpu_id=args.get('gpu_id', 0),
            script_path=args.get('script_path'),
        )


