"""
Wrapper tool for RDiff_seq sequence design.
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


def _rdiff_seq_path() -> str:
    return require_env_value("RDIFF_SEQ_CODE_DIR")


def _rdiff_python() -> str:
    return require_env_value("PYTHON_CONDA_RDIFF")


def _offset_header(header: str, offset: int) -> str:
    """Offset trailing numeric sample index while preserving backbone stem."""
    if offset == 0:
        return header
    parts = header.split(maxsplit=1)
    name = parts[0]
    suffix = f" {parts[1]}" if len(parts) > 1 else ""
    if "_" in name:
        prefix, last = name.rsplit("_", 1)
        if last.isdigit():
            return f"{prefix}_{int(last) + offset}{suffix}"
    return f"{name}_{offset}{suffix}"


def _read_fasta_records(path: str) -> list[tuple[str, str]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records: list[tuple[str, str]] = []
    header = None
    seq_parts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts)))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def _write_split_fastas(records: list[tuple[str, str]], output_dir: str, *, header_offset: int) -> list[str]:
    written: list[str] = []
    for header, sequence in records:
        final_header = _offset_header(header, header_offset)
        file_name = final_header.split()[0] + ".fasta"
        path = os.path.join(output_dir, file_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f">{final_header}\n{sequence}")
        written.append(path)
    return written


def _run_sequence_design_once(
    *,
    ss_db_constrain: str,
    input_structure_dir: str,
    output_dir: str,
    num_samples: int,
    fixed_seq: Optional[str],
    target_seq: Optional[str],
    model_path: str,
    batch_size: int,
    gpu_id: int,
    run_tag: str,
) -> dict:
    dir_name = os.path.basename(input_structure_dir) + "_seq"
    shard_dir = os.path.join(output_dir, f"checkpoint_{run_tag}")
    os.makedirs(shard_dir, exist_ok=True)

    output_fasta = f"{dir_name}_{run_tag}.fasta"
    final_fasta_path = os.path.join(shard_dir, output_fasta)
    rdiff_seq_path = _rdiff_seq_path()
    python_bin = _rdiff_python()
    cmd = [
        python_bin,
        "main.py",
        "--config", "configs/ribodiffusion_af3_ss.py",
        "--no_wandb",
        "--mode", "inf_dir",
        f"--inf_folder={input_structure_dir}",
        f"--config.eval.output_fasta_file={final_fasta_path}",
        f"--config.eval.batch_size={batch_size}",
        f"--config.eval.n_samples={num_samples}",
        "--config.data.ss_channel=1",
        f"--config.eval.model_path={model_path}",
        f"--config.eval.input_db={ss_db_constrain}",
    ]

    if fixed_seq:
        cmd.append("--config.eval.fixed_part=True")
        cmd.append(f"--config.eval.fixed_seq={fixed_seq}")
    else:
        cmd.append("--config.eval.fixed_part=False")

    if target_seq:
        cmd.append(f"--config.eval.target_seq={target_seq}")
        cmd.append("--config.eval.partial_sampling=True")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{rdiff_seq_path}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    result = subprocess.run(
        cmd,
        cwd=rdiff_seq_path,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        return {
            "success": False,
            "model_path": model_path,
            "num_samples": num_samples,
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    if not os.path.exists(final_fasta_path):
        return {
            "success": False,
            "model_path": model_path,
            "num_samples": num_samples,
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "error": f"Expected FASTA not found: {final_fasta_path}",
        }
    records = _read_fasta_records(final_fasta_path)
    return {
        "success": True,
        "model_path": model_path,
        "num_samples": num_samples,
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "shard_fasta": final_fasta_path,
        "records": records,
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-1000:],
    }


def generate_rna_sequences(
    ss_db_constrain: str,
    input_structure_dir: str,
    output_dir: Optional[str] = None,
    num_samples: int = 10,
    fixed_seq: Optional[str] = None,
    target_seq: Optional[str] = None,
    model_path: Optional[str] = None,
    batch_size: int = 10,
    gpu_id: int = 0,
    checkpoint_dir: Optional[str] = None,
    checkpoint_paths: Optional[list[str] | str] = None,
) -> str:
    """
    Run RDiff_seq inference to generate RNA sequences from secondary structure.
    
    Args:
        ss_db_constrain: Dot-bracket constraint string (secondary structure)
        output_dir: Output directory for generated sequences
        num_samples: Number of samples to generate
        fixed_seq: Optional fixed sequence motif (e.g. '--------------UAC-------------GCA')
        target_seq: Optional target sequence for partial sampling
        model_path: Path to model checkpoint
        batch_size: Inference batch size
        
    Returns:
        JSON string with execution results
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        checkpoints = resolve_checkpoint_paths(
            checkpoint_dir=checkpoint_dir,
            checkpoint_paths=checkpoint_paths,
            env_dir_vars="RDIFF_SEQUENCE_CHECKPOINT_DIR",
            fallback_path=model_path,
        )
        if not checkpoints:
            return json.dumps({
                "success": False,
                "error": "No sequence design checkpoints found. Set RDIFF_SEQUENCE_CHECKPOINT_DIR in .env or pass checkpoint_dir/checkpoint_paths.",
            }, ensure_ascii=False)

        sample_splits = split_samples(num_samples, len(checkpoints))
        shard_results = []
        written_fastas: list[str] = []
        all_records: list[tuple[str, str]] = []
        offset = 0
        for idx, (ckpt, shard_samples) in enumerate(zip(checkpoints, sample_splits), start=1):
            if shard_samples <= 0:
                continue
            tag = checkpoint_tag(ckpt, idx)
            shard = _run_sequence_design_once(
                ss_db_constrain=ss_db_constrain,
                input_structure_dir=input_structure_dir,
                output_dir=output_dir,
                num_samples=shard_samples,
                fixed_seq=fixed_seq,
                target_seq=target_seq,
                model_path=str(ckpt),
                batch_size=batch_size,
                gpu_id=gpu_id,
                run_tag=tag,
            )
            shard_results.append({k: v for k, v in shard.items() if k != "records"})
            if not shard.get("success"):
                return json.dumps({
                    "success": False,
                    "error": "One checkpoint shard failed",
                    "failed_shard": shard_results[-1],
                    "all_shards": shard_results,
                }, ensure_ascii=False)
            records = shard["records"]
            written_fastas.extend(_write_split_fastas(records, output_dir, header_offset=offset))
            all_records.extend((_offset_header(h, offset), s) for h, s in records)
            offset += shard_samples

        return json.dumps({
            "success": True,
            "output_dir": output_dir,
            "requested_num_samples": num_samples,
            "checkpoint_count": len(checkpoints),
            "checkpoints": [str(p) for p in checkpoints],
            "sample_splits": sample_splits,
            "written_fasta_count": len(written_fastas),
            "written_fastas": written_fastas[:20],
            "sequences_preview": [seq for _, seq in all_records[:10]],
            "shards": shard_results,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@register_tool("rdiff_seq_design_tool")
class RDiffSeqDesignTool(BaseTool):
    """Tool for RNA sequence design using RDiff_seq"""

    description = 'Design RNA sequences based on secondary structure constraints and input structures using RDiff_seq.'
    parameters = {
        'type': 'object',
        'properties': {
            'ss_db_constrain': {
                'type': 'string',
                'description': 'RNA secondary structure constraint (dot-bracket format string)'
            },
            'input_structure_dir': {
                'type': 'string',
                'description': 'Input structure directory path (PDB format)'
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
            'fixed_seq': {
                'type': 'string',
                'description': 'Fixed sequence motif (optional), e.g., "--------------UAC-------------GCA", where "-" indicates variable positions'
            },
            'target_seq': {
                'type': 'string',
                'description': 'Target sequence (optional), used for partial sampling mode'
            },
            'batch_size': {
                'type': 'integer',
                'default': 10,
                'description': 'Batch size for inference'
            },
            'gpu_id': {
                'type': 'integer',
                'default': 0,
                'description': 'GPU ID to use'
            },
            'checkpoint_dir': {
                'type': 'string',
                'description': 'Optional checkpoint directory. If omitted, read RDIFF_SEQUENCE_CHECKPOINT_DIR from .env/environment.'
            },
            'checkpoint_paths': {
                'type': 'array',
                'description': 'Optional explicit checkpoint paths. Overrides checkpoint_dir.'
            }
        },
        'required': ['ss_db_constrain', 'input_structure_dir', 'output_dir','fixed_seq']
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return generate_rna_sequences(
            ss_db_constrain=args['ss_db_constrain'],
            input_structure_dir=args['input_structure_dir'],
            output_dir=args['output_dir'],
            num_samples=args.get('num_samples', 10),
            fixed_seq=args.get('fixed_seq'),
            target_seq=args.get('target_seq'),
            batch_size=args.get('batch_size', 10),
            gpu_id=args.get('gpu_id', 0),
            checkpoint_dir=args.get('checkpoint_dir'),
            checkpoint_paths=args.get('checkpoint_paths'),
        )

