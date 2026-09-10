"""
Wrapper tool for RDiff_str backbone generation.
"""
import os
import json
import subprocess
from pathlib import Path
from typing import Optional

from rhoagent.tools.base import BaseTool, register_tool
from .checkpoint_utils import (
    checkpoint_tag,
    require_env_value,
    resolve_checkpoint_paths,
    split_samples,
)


def _rdiff_str_path() -> str:
    return require_env_value("RDIFF_STR_CODE_DIR")


def _rdiff_python() -> str:
    return require_env_value("PYTHON_CONDA_RDIFF")


def _run_backbone_once(
    *,
    ss_db_constrain: str,
    output_dir: str,
    num_samples: int,
    gpu_id: int,
    seed: int,
    num_t: int,
    weights_path: str,
    run_name: str,
) -> dict:
    target_length = len(ss_db_constrain)
    rdiff_str_path = _rdiff_str_path()
    python_bin = _rdiff_python()
    cmd = [
        python_bin,
        "inference_3d_ribodiffusion.py",
        "--config-name=rna_cww_ss_training",
        f"inference.output_dir={output_dir}",
        f"inference.weights_path={weights_path}",
        f"inference.num_samples={num_samples}",
        f"inference.gpu_id={gpu_id}",
        f"inference.seed={seed}",
        f"inference.diffusion.num_t={num_t}",
        "inference.task_name=ss_conditional",
        f"inference.target_length={target_length}",
        f"inference.ss_db_constrain='{ss_db_constrain}'",
        f"inference.name={run_name}",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{rdiff_str_path}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        cmd,
        cwd=rdiff_str_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )

    final_output_dir = os.path.join(output_dir, run_name)
    pdb_files = []
    if os.path.exists(final_output_dir):
        for file_name in os.listdir(final_output_dir):
            if file_name.endswith(".pdb"):
                pdb_files.append(os.path.join(final_output_dir, file_name))

    return {
        "success": result.returncode == 0,
        "output_dir": final_output_dir,
        "generated_files": pdb_files,
        "pdb_count": len(pdb_files),
        "weights_path": weights_path,
        "num_samples": num_samples,
        "seed": seed,
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:] if result.returncode != 0 else result.stdout[-1000:],
        "stderr": result.stderr[-4000:] if result.returncode != 0 else result.stderr[-1000:],
    }


def generate_rna_backbone(
    ss_db_constrain: str,
    output_dir: Optional[str] = None,
    num_samples: int = 10,
    gpu_id: int = 0,
    seed: int = 42,
    num_t: int = 500,
    checkpoint_dir: Optional[str] = None,
    checkpoint_paths: Optional[list[str] | str] = None,
) -> str:
    """
    Run RDiff_str inference to generate RNA backbones.
    
    Args:
        ss_db_constrain: Dot-bracket constraint string
        output_dir: Output directory for generated PDBs
        num_samples: Number of samples to generate
        gpu_id: GPU ID to use
        seed: Random seed
        num_t: Diffusion steps
        
    Returns:
        JSON string with execution results
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "ribodiffusion_output")
    
    os.makedirs(output_dir, exist_ok=True)
    try:
        checkpoints = resolve_checkpoint_paths(
            checkpoint_dir=checkpoint_dir,
            checkpoint_paths=checkpoint_paths,
            env_dir_vars="RDIFF_BACKBONE_CHECKPOINT_DIR",
            fallback_path=None,
        )
        if not checkpoints:
            return json.dumps({
                "success": False,
                "error": "No backbone checkpoints found. Set RDIFF_BACKBONE_CHECKPOINT_DIR in .env or pass checkpoint_dir/checkpoint_paths.",
            }, ensure_ascii=False)

        sample_splits = split_samples(num_samples, len(checkpoints))
        target_length = len(ss_db_constrain)
        shard_results = []
        generated_files = []
        for idx, (ckpt, shard_samples) in enumerate(zip(checkpoints, sample_splits), start=1):
            if shard_samples <= 0:
                continue
            tag = checkpoint_tag(ckpt, idx)
            run_name = f"gen_len{target_length}_{num_samples}samples_{tag}"
            shard = _run_backbone_once(
                ss_db_constrain=ss_db_constrain,
                output_dir=output_dir,
                num_samples=shard_samples,
                gpu_id=gpu_id,
                seed=seed + idx - 1,
                num_t=num_t,
                weights_path=str(ckpt),
                run_name=run_name,
            )
            shard_results.append(shard)
            generated_files.extend(shard.get("generated_files", []))
            if not shard.get("success"):
                return json.dumps({
                    "success": False,
                    "error": "One backbone checkpoint shard failed",
                    "failed_shard": shard,
                    "all_shards": shard_results,
                }, ensure_ascii=False)

        return json.dumps({
            "success": True,
            "output_dir": output_dir,
            "generated_files": generated_files,
            "pdb_count": len(generated_files),
            "requested_num_samples": num_samples,
            "checkpoint_count": len(checkpoints),
            "checkpoints": [str(p) for p in checkpoints],
            "sample_splits": sample_splits,
            "shards": shard_results,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@register_tool("rdiff_3d_backbone_design_tool")
class RDiff3DBackboneDesignTool(BaseTool):
    """Tool for RNA backbone generation using RDiff_str"""

    description = 'Generate RNA 3D backbone structures (PDB format) based on secondary structure constraints using RDiff_str.'
    parameters = {
        'type': 'object',
        'properties': {
            'ss_db_constrain': {
                'type': 'string',
                'description': 'RNA secondary structure constraint (dot-bracket format string)'
            },
            'output_dir': {
                'type': 'string',
                'description': 'Output directory for results'
            },
            'num_samples': {
                'type': 'integer',
                'default': 10,
                'description': 'Number of samples to generate'
            },
            'gpu_id': {
                'type': 'integer',
                'default': 0,
                'description': 'GPU ID to use'
            },
            'seed': {
                'type': 'integer',
                'default': 42,
                'description': 'Random seed'
            },
            'checkpoint_dir': {
                'type': 'string',
                'description': 'Optional checkpoint directory. If omitted, read RDIFF_BACKBONE_CHECKPOINT_DIR from .env/environment.'
            },
            'checkpoint_paths': {
                'type': 'array',
                'description': 'Optional explicit checkpoint paths. Overrides checkpoint_dir.'
            }
        },
        'required': ['ss_db_constrain', 'output_dir']
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return generate_rna_backbone(
            ss_db_constrain=args['ss_db_constrain'],
            output_dir=args['output_dir'],
            num_samples=args.get('num_samples', 10),
            gpu_id=args.get('gpu_id', 0),
            seed=args.get('seed', 42),
            checkpoint_dir=args.get('checkpoint_dir'),
            checkpoint_paths=args.get('checkpoint_paths'),
        )

