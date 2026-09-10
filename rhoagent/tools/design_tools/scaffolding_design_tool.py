"""
Wrapper tool for RDiff_str motif scaffolding.

This tool is for Broccoli/Pepper/FRET-style motif scaffolding where RDiff_str
uses a motif PDB and contig specification to keep motif frames fixed while
sampling scaffold residues. It wraps:

  inference_3d_ribodiffusion.py --config-name=rna_scaffolding_training
  inference.task_name=scaffolding_iter
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

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


def _count_outputs(output_dir: Path) -> dict[str, int]:
    if not output_dir.is_dir():
        return {"pdb_count": 0, "fasta_count": 0}
    return {
        "pdb_count": sum(1 for _ in output_dir.rglob("*.pdb")),
        "fasta_count": sum(1 for _ in output_dir.rglob("*.fasta")),
    }


def _copy_scaffold_outputs(src_dir: Path, dst_dir: Path, prefix: str) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(src_dir.glob("*")):
        if path.suffix.lower() not in {".pdb", ".fasta", ".fa"}:
            continue
        dst = dst_dir / f"{prefix}_{path.name}"
        shutil.copy2(path, dst)
        copied.append(str(dst))
    return copied


def run_rdiff_scaffolding(
    *,
    target_pdb_path: str,
    contigs: str,
    output_dir: str,
    inference_name: str = "scaffolding_iter",
    target_length: Optional[int] = None,
    length_min: int = 120,
    length_max: int = 120,
    length_step: int = 25,
    max_contigs_per_length: int = 200,
    num_samples: int = 5,
    gpu_id: int = 0,
    seed: int = 42,
    diffusion_num_t: Optional[int] = None,
    weights_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_paths: Optional[list[str] | str] = None,
    python_bin: Optional[str] = None,
    rdiff_str_dir: Optional[str] = None,
    extra_overrides: Optional[list[str]] = None,
) -> str:
    """Run RDiff_str motif scaffolding and return a JSON-formatted summary."""
    target_pdb = Path(target_pdb_path).resolve()
    out_root = Path(output_dir).resolve()
    python_bin = python_bin or _rdiff_python()
    rdiff_dir = Path(rdiff_str_dir or _rdiff_str_path()).resolve()

    if not target_pdb.is_file():
        return json.dumps(
            {"success": False, "error": f"target_pdb_path not found: {target_pdb}"},
            ensure_ascii=False,
        )
    if not rdiff_dir.is_dir():
        return json.dumps(
            {"success": False, "error": f"RDiff_str directory not found: {rdiff_dir}"},
            ensure_ascii=False,
        )
    if not contigs:
        return json.dumps({"success": False, "error": "contigs is required"}, ensure_ascii=False)

    out_root.mkdir(parents=True, exist_ok=True)
    merged_scaffold_dir = out_root / inference_name

    checkpoints = resolve_checkpoint_paths(
        checkpoint_dir=checkpoint_dir,
        checkpoint_paths=checkpoint_paths,
        env_dir_vars="RDIFF_SCAFFOLDING_CHECKPOINT_DIR",
        fallback_path=weights_path,
    )
    if not checkpoints:
        return json.dumps(
            {
                "success": False,
                "error": "No scaffolding checkpoints found. Set RDIFF_SCAFFOLDING_CHECKPOINT_DIR in .env or pass checkpoint_dir/checkpoint_paths.",
            },
            ensure_ascii=False,
        )

    sample_splits = split_samples(num_samples, len(checkpoints))
    shard_results: list[dict[str, Any]] = []
    copied_files: list[str] = []

    try:
        for idx, (checkpoint, shard_samples) in enumerate(zip(checkpoints, sample_splits), start=1):
            if shard_samples <= 0:
                continue
            tag = checkpoint_tag(checkpoint, idx)
            shard_inference_name = f"{inference_name}_{tag}"
            shard_dir = out_root / shard_inference_name
            cmd = [
                python_bin,
                "inference_3d_ribodiffusion.py",
                "--config-name=rna_scaffolding_training",
                "inference.task_name=scaffolding_iter",
                f"inference.name={shard_inference_name}",
                f"inference.weights_path={checkpoint}",
                f"inference.target_pdb_path={target_pdb}",
                f"inference.output_dir={out_root}",
                f"inference.num_samples={shard_samples}",
                f"inference.seed={seed + idx - 1}",
                f"inference.contigs={contigs}",
                f"inference.scaffold_iter.length_min={length_min}",
                f"inference.scaffold_iter.length_max={length_max}",
                f"inference.scaffold_iter.length_step={length_step}",
                f"inference.scaffold_iter.max_contigs_per_length={max_contigs_per_length}",
                f"inference.gpu_id={gpu_id}",
            ]
            if target_length is not None:
                cmd.append(f"inference.target_length={target_length}")
            if diffusion_num_t is not None:
                cmd.append(f"inference.diffusion.num_t={diffusion_num_t}")
            if extra_overrides:
                cmd.extend(extra_overrides)

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{rdiff_dir}:{env.get('PYTHONPATH', '')}"
            result = subprocess.run(
                cmd,
                cwd=str(rdiff_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                env=env,
            )
            counts = _count_outputs(shard_dir)
            shard_payload: dict[str, Any] = {
                "success": result.returncode == 0 and shard_dir.is_dir(),
                "checkpoint": str(checkpoint),
                "num_samples": shard_samples,
                "seed": seed + idx - 1,
                "inference_name": shard_inference_name,
                "scaffold_dir": str(shard_dir),
                "command": " ".join(cmd),
                "returncode": result.returncode,
                **counts,
            }
            if result.returncode != 0:
                shard_payload["error"] = f"RDiff_str scaffolding failed with exit code {result.returncode}"
                shard_payload["stdout"] = result.stdout[-4000:]
                shard_payload["stderr"] = result.stderr[-4000:]
                shard_results.append(shard_payload)
                return json.dumps(
                    {
                        "success": False,
                        "error": "One scaffolding checkpoint shard failed",
                        "failed_shard": shard_payload,
                        "shards": shard_results,
                    },
                    ensure_ascii=False,
                )
            copied_files.extend(_copy_scaffold_outputs(shard_dir, merged_scaffold_dir, tag))
            shard_results.append(shard_payload)

        counts = _count_outputs(merged_scaffold_dir)
        payload: dict[str, Any] = {
            "success": True,
            "target_pdb_path": str(target_pdb),
            "contigs": contigs,
            "output_root": str(out_root),
            "scaffold_dir": str(merged_scaffold_dir),
            "inference_name": inference_name,
            "length_min": length_min,
            "length_max": length_max,
            "length_step": length_step,
            "max_contigs_per_length": max_contigs_per_length,
            "requested_num_samples": num_samples,
            "checkpoint_count": len(checkpoints),
            "checkpoints": [str(p) for p in checkpoints],
            "sample_splits": sample_splits,
            "gpu_id": gpu_id,
            "seed": seed,
            "copied_files": copied_files[:50],
            "shards": shard_results,
            **counts,
        }
        if target_length is not None:
            payload["target_length"] = target_length
        if diffusion_num_t is not None:
            payload["diffusion_num_t"] = diffusion_num_t
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "success": False,
                "error": str(exc),
                "target_pdb_path": str(target_pdb),
                "contigs": contigs,
                "output_root": str(out_root),
                "shards": shard_results,
            },
            ensure_ascii=False,
        )


@register_tool("rdiff_scaffolding_design_tool")
class RDiffScaffoldingDesignTool(BaseTool):
    """Tool for RDiff_str motif scaffolding with target_pdb_path + contigs."""

    description = (
        "Run RDiff_str motif scaffolding using rna_scaffolding_training and "
        "scaffolding_iter. Use this for Broccoli/Pepper/FRET-style motif scaffolding, "
        "where target_pdb_path and contigs define fixed motif residues and generated "
        "scaffold gaps."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target_pdb_path": {
                "type": "string",
                "description": "Motif/reference PDB path used by RDiff_str scaffolding.",
            },
            "contigs": {
                "type": "string",
                "description": "RDiff_str contig specification, e.g. '8+/A3-16/15+/A29-47/8+'.",
            },
            "output_dir": {
                "type": "string",
                "description": "Output root directory. Final PDB/FASTA files are under output_dir/inference_name.",
            },
            "inference_name": {
                "type": "string",
                "default": "scaffolding_iter",
                "description": "RDiff_str inference.name subdirectory.",
            },
            "target_length": {
                "type": "integer",
                "description": "Optional fixed target length for scaffolding.",
            },
            "length_min": {
                "type": "integer",
                "default": 120,
                "description": "Minimum target length for scaffolding_iter sweep.",
            },
            "length_max": {
                "type": "integer",
                "default": 120,
                "description": "Maximum target length for scaffolding_iter sweep.",
            },
            "length_step": {
                "type": "integer",
                "default": 25,
                "description": "Length step for scaffolding_iter sweep.",
            },
            "max_contigs_per_length": {
                "type": "integer",
                "default": 200,
                "description": "Maximum concrete contigs sampled per length.",
            },
            "num_samples": {
                "type": "integer",
                "default": 5,
                "description": "Diffusion samples per concrete contig.",
            },
            "gpu_id": {
                "type": "integer",
                "default": 0,
                "description": "GPU ID passed to inference.gpu_id.",
            },
            "seed": {
                "type": "integer",
                "default": 42,
                "description": "Random seed for contig subsampling and diffusion.",
            },
            "diffusion_num_t": {
                "type": "integer",
                "description": "Optional diffusion step override, maps to inference.diffusion.num_t.",
            },
            "weights_path": {
                "type": "string",
                "description": "Fallback single RDiff_str scaffolding checkpoint path.",
            },
            "checkpoint_dir": {
                "type": "string",
                "description": "Optional checkpoint directory. If omitted, read RDIFF_SCAFFOLDING_CHECKPOINT_DIR from .env/environment.",
            },
            "checkpoint_paths": {
                "type": "array",
                "description": "Optional explicit checkpoint paths. Overrides checkpoint_dir.",
            },
            "python_bin": {
                "type": "string",
                "description": "Optional Python executable for RDiff_str environment.",
            },
            "rdiff_str_dir": {
                "type": "string",
                "description": "Optional RDiff_str repository directory.",
            },
            "extra_overrides": {
                "type": "array",
                "description": "Optional extra Hydra overrides, each item is a string.",
            },
        },
        "required": ["target_pdb_path", "contigs", "output_dir"],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return run_rdiff_scaffolding(
            target_pdb_path=args["target_pdb_path"],
            contigs=args["contigs"],
            output_dir=args["output_dir"],
            inference_name=args.get("inference_name", "scaffolding_iter"),
            target_length=args.get("target_length"),
            length_min=args.get("length_min", 120),
            length_max=args.get("length_max", 120),
            length_step=args.get("length_step", 25),
            max_contigs_per_length=args.get("max_contigs_per_length", 200),
            num_samples=args.get("num_samples", 5),
            gpu_id=args.get("gpu_id", 0),
            seed=args.get("seed", 42),
            diffusion_num_t=args.get("diffusion_num_t"),
            weights_path=args.get("weights_path"),
            checkpoint_dir=args.get("checkpoint_dir"),
            checkpoint_paths=args.get("checkpoint_paths"),
            python_bin=args.get("python_bin"),
            rdiff_str_dir=args.get("rdiff_str_dir"),
            extra_overrides=args.get("extra_overrides"),
        )
