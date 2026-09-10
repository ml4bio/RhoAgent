"""
Scaffolding-specific structure evaluation tool.

This evaluator follows the scaffolding pipeline logic:
1. Optionally check whether AF3 can reproduce the wild-type motif first.
2. If the WT motif check fails, skip motif RMSD/TM filtering for designed
   scaffolds and evaluate only self-consistency, compactness, and unintended
   motif-scaffold contacts.
3. If the WT motif check passes, evaluate designed AF3 predictions for motif
   preservation in addition to backbone self-consistency and compactness.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rhoagent.tools.base import BaseTool, register_tool
from rhoagent.tools.eval_tools.utils.eval_utils import (
    DEFAULT_USALIGN_BIN,
    evaluate_structure,
    extract_fixed_motif_indices,
    kabsch_align,
    run_usalign,
)
from rhoagent.tools.eval_tools.utils.pdb_utils import _get_parser, _residue_to_base


def read_first_sequence(fasta_path: Path) -> str:
    lines = fasta_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if line and not line.startswith(">"):
            return line.upper().replace("T", "U")
    raise ValueError(f"No sequence line found in {fasta_path}")


def infer_backbone_stem(pred_name: str) -> str:
    """e.g. name ending with sequence index ..._0_24 -> backbone stem ..._0."""
    name = pred_name.replace("_model", "")
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[0]


def collect_pred_structures(pred_dir: Path) -> list[Path]:
    if pred_dir.is_file():
        return [pred_dir]
    structures: list[Path] = []
    for sub in sorted(pred_dir.iterdir()):
        if sub.is_dir():
            cifs = sorted(sub.glob("*_model.cif")) or sorted(sub.glob("*.cif"))
            pdbs = sorted(sub.glob("*.pdb"))
            if cifs:
                structures.append(cifs[0])
            elif pdbs:
                structures.append(pdbs[0])
        elif sub.suffix.lower() in {".cif", ".mmcif", ".pdb"}:
            structures.append(sub)
    return structures


def find_backbone_files(backbone_dir: Path, stem: str) -> tuple[Optional[Path], Optional[Path]]:
    stem_lower = stem.lower()
    pdb_path: Optional[Path] = None
    fasta_path: Optional[Path] = None
    for path in backbone_dir.iterdir():
        if not path.is_file() or path.stem.lower() != stem_lower:
            continue
        if path.suffix.lower() == ".pdb":
            pdb_path = path
        elif path.suffix.lower() in {".fa", ".fas", ".fasta"}:
            fasta_path = path
    return pdb_path, fasta_path


def extract_c1_prime_coords_ordered(structure_path: str) -> tuple[np.ndarray, str]:
    parser = _get_parser(Path(structure_path).suffix)
    structure = parser.get_structure("X", structure_path)

    coords: list[np.ndarray] = []
    seq_chars: list[str] = []
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            base = _residue_to_base(residue)
            if base is None:
                continue
            c1 = None
            for atom_name in ("C1'", "C1*"):
                if atom_name in residue:
                    c1 = residue[atom_name].get_coord()
                    break
            if c1 is None:
                continue
            coords.append(np.asarray(c1, dtype=float))
            seq_chars.append(base)

    if not coords:
        raise ValueError(f"No RNA C1' atoms found in {structure_path}")
    return np.vstack(coords), "".join(seq_chars)


def parse_indices(indices: list[int], *, one_based: bool) -> set[int]:
    if one_based:
        return {i - 1 for i in indices}
    return set(indices)


def motif_indices_from_spec(spec: dict[str, Any], fallback_fixed_motif: str = "") -> set[int]:
    if spec.get("motif_indices"):
        return parse_indices(list(spec["motif_indices"]), one_based=spec.get("indices_one_based", True))
    fixed_motif = spec.get("fixed_motif") or fallback_fixed_motif
    if not fixed_motif:
        raise ValueError("Motif spec needs motif_indices, fixed_motif, or backbone FASTA fixed motif.")
    return set(extract_fixed_motif_indices(fixed_motif))


def subset_coords(structure_path: str, indices: set[int]) -> tuple[np.ndarray, str]:
    coords, seq = extract_c1_prime_coords_ordered(structure_path)
    ordered = sorted(indices)
    if not ordered:
        raise ValueError("Empty motif index set")
    if ordered[-1] >= len(coords):
        raise ValueError(f"Motif index {ordered[-1]} out of range for {structure_path} length={len(coords)}")
    return coords[ordered], "".join(seq[i] for i in ordered)


def write_c1_pdb(coords: np.ndarray, seq: str, out_path: Path) -> None:
    lines: list[str] = []
    for atom_idx, (coord, base) in enumerate(zip(coords, seq), start=1):
        x, y, z = coord
        lines.append(
            f"ATOM  {atom_idx:5d}  C1' {base:>3} A{atom_idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def eval_motif_alignment(
    *,
    pred_path: str,
    motif_ref_pdb: str,
    motif_indices: set[int],
    usalign_bin: str,
) -> dict[str, Any]:
    ref_coords, ref_seq = extract_c1_prime_coords_ordered(motif_ref_pdb)
    pred_coords, pred_seq = subset_coords(pred_path, motif_indices)
    n = min(len(ref_coords), len(pred_coords))
    if n < 2:
        raise ValueError("Need at least 2 motif residues for motif alignment")
    ref_coords = ref_coords[:n]
    pred_coords = pred_coords[:n]
    ref_seq = ref_seq[:n]
    pred_seq = pred_seq[:n]

    _, rmsd = kabsch_align(ref_coords, pred_coords)
    tm_score: Optional[float] = None
    with tempfile.TemporaryDirectory() as tmp:
        ref_tmp = Path(tmp) / "ref_motif.pdb"
        pred_tmp = Path(tmp) / "pred_motif.pdb"
        write_c1_pdb(ref_coords, ref_seq, ref_tmp)
        write_c1_pdb(pred_coords, pred_seq, pred_tmp)
        tm_score, _ = run_usalign(str(ref_tmp), str(pred_tmp), usalign_bin=usalign_bin)

    return {
        "motif_rmsd": float(rmsd),
        "motif_tm_score": tm_score,
        "motif_seq_match": ref_seq == pred_seq,
        "motif_ref_len": len(ref_seq),
        "motif_pred_len": len(pred_seq),
    }


def compute_lddt(ref_coords: np.ndarray, pred_coords: np.ndarray) -> tuple[Optional[float], list[Optional[float]]]:
    n = min(len(ref_coords), len(pred_coords))
    if n < 2:
        return None, []
    ref = ref_coords[:n]
    pred = pred_coords[:n]
    per_residue: list[Optional[float]] = []
    thresholds = (0.5, 1.0, 2.0, 4.0)

    for i in range(n):
        scores: list[float] = []
        for j in range(n):
            if i == j:
                continue
            d_ref = float(np.linalg.norm(ref[i] - ref[j]))
            if d_ref >= 15.0:
                continue
            d_pred = float(np.linalg.norm(pred[i] - pred[j]))
            delta = abs(d_pred - d_ref)
            scores.append(sum(1.0 for t in thresholds if delta < t) / len(thresholds))
        per_residue.append(float(np.mean(scores)) if scores else None)

    valid = [v for v in per_residue if v is not None]
    return (float(np.mean(valid)) if valid else None), per_residue


def compute_compactness_metrics(coords: np.ndarray, motif_indices: set[int]) -> dict[str, Any]:
    n = len(coords)
    centroid = coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
    non_motif = sorted(i for i in range(n) if i not in motif_indices)

    long_range_contacts = 0
    long_range_pairs = 0
    for pos, i in enumerate(non_motif):
        for j in non_motif[pos + 1 :]:
            if j - i < 10:
                continue
            long_range_pairs += 1
            if float(np.linalg.norm(coords[i] - coords[j])) < 12.0:
                long_range_contacts += 1

    isolated = 0
    for i in non_motif:
        contacts = 0
        for j in non_motif:
            if i == j or abs(j - i) < 5:
                continue
            if float(np.linalg.norm(coords[i] - coords[j])) < 15.0:
                contacts += 1
        if contacts < 2:
            isolated += 1

    return {
        "rg": rg,
        "rg_norm": rg / math.sqrt(n),
        "end_to_end_dist": float(np.linalg.norm(coords[0] - coords[-1])),
        "end_to_end_dist_norm": float(np.linalg.norm(coords[0] - coords[-1])) / n,
        "non_motif_long_range_contact_density": (
            long_range_contacts / long_range_pairs if long_range_pairs else None
        ),
        "non_motif_long_range_contacts": long_range_contacts,
        "non_motif_long_range_pairs": long_range_pairs,
        "isolated_scaffold_fraction": isolated / len(non_motif) if non_motif else None,
        "non_motif_residue_count": len(non_motif),
    }


def compute_motif_scaffold_contacts(
    coords: np.ndarray,
    motif_indices: set[int],
    *,
    distance_cutoff: float = 8.0,
    min_seq_sep: int = 8,
) -> int:
    motif = sorted(i for i in motif_indices if i < len(coords))
    non_motif = [i for i in range(len(coords)) if i not in motif_indices]
    count = 0
    for i in motif:
        for j in non_motif:
            if abs(i - j) < min_seq_sep:
                continue
            if float(np.linalg.norm(coords[i] - coords[j])) < distance_cutoff:
                count += 1
    return count


def run_wt_motif_gate(
    *,
    motif_specs: list[dict[str, Any]],
    usalign_bin: str,
    rmsd_threshold: float,
    tm_threshold: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if not motif_specs:
        return {"enabled": False, "passed": False, "reason": "no motif_specs provided", "checks": checks}

    usable_specs = [s for s in motif_specs if s.get("wt_pred_path") and s.get("motif_ref_pdb")]
    if not usable_specs:
        return {
            "enabled": False,
            "passed": False,
            "reason": "no WT motif prediction inputs provided",
            "checks": checks,
        }

    passed = True
    for spec in usable_specs:
        try:
            wt_indices = motif_indices_from_spec(
                {
                    "motif_indices": spec.get("wt_motif_indices"),
                    "indices_one_based": spec.get("wt_indices_one_based", True),
                    "fixed_motif": spec.get("wt_fixed_motif"),
                }
            )
            metrics = eval_motif_alignment(
                pred_path=spec["wt_pred_path"],
                motif_ref_pdb=spec["motif_ref_pdb"],
                motif_indices=wt_indices,
                usalign_bin=usalign_bin,
            )
            ok = (
                metrics["motif_rmsd"] <= rmsd_threshold
                and (metrics["motif_tm_score"] is None or metrics["motif_tm_score"] >= tm_threshold)
            )
            checks.append({"name": spec.get("name", "motif"), "passed": ok, **metrics})
            if not ok:
                passed = False
        except Exception as exc:  # noqa: BLE001
            passed = False
            checks.append({"name": spec.get("name", "motif"), "passed": False, "error": str(exc)})

    return {
        "enabled": True,
        "passed": passed,
        "reason": "WT motif check passed" if passed else "WT motif check failed; motif evaluation should be disabled",
        "thresholds": {"motif_rmsd_max": rmsd_threshold, "motif_tm_min": tm_threshold},
        "checks": checks,
    }


def write_csv(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: set[str] = set()
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                flat[key] = ""
            elif isinstance(value, (list, dict)):
                flat[key] = json.dumps(value, ensure_ascii=False)
            else:
                flat[key] = value
        flat_rows.append(flat)
        fieldnames.update(flat)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(flat_rows)


def scaffolding_structure_eval(
    *,
    pred_dir: str,
    backbone_dir: str,
    output_dir: str,
    motif_specs: Optional[list[dict[str, Any]]] = None,
    enable_motif_eval: str = "auto",
    wt_motif_rmsd_threshold: float = 4.0,
    wt_motif_tm_threshold: float = 0.45,
    usalign_bin: str = DEFAULT_USALIGN_BIN,
    max_predictions: Optional[int] = None,
) -> dict[str, Any]:
    pred_root = Path(pred_dir).resolve()
    backbone_root = Path(backbone_dir).resolve()
    out_root = Path(output_dir).resolve()
    output_csv = out_root / "results.csv"
    gate_path = out_root / "wt_motif_gate.json"

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")
    if not backbone_root.exists():
        raise FileNotFoundError(f"Backbone directory not found: {backbone_root}")

    motif_specs = motif_specs or []
    if enable_motif_eval not in {"auto", "true", "false"}:
        raise ValueError('enable_motif_eval must be one of "auto", "true", "false"')

    wt_gate = run_wt_motif_gate(
        motif_specs=motif_specs,
        usalign_bin=usalign_bin,
        rmsd_threshold=wt_motif_rmsd_threshold,
        tm_threshold=wt_motif_tm_threshold,
    )
    gate_by_name = {check.get("name"): bool(check.get("passed")) for check in wt_gate.get("checks", [])}

    def is_motif_eval_enabled(spec: dict[str, Any], idx: int) -> bool:
        if enable_motif_eval == "true":
            return True
        if enable_motif_eval == "false":
            return False
        name = spec.get("name") or f"motif_{idx}"
        return bool(gate_by_name.get(name, False))

    motif_eval_enabled_by_motif = {
        spec.get("name") or f"motif_{idx}": is_motif_eval_enabled(spec, idx)
        for idx, spec in enumerate(motif_specs, start=1)
    }
    motif_eval_enabled = any(motif_eval_enabled_by_motif.values())

    out_root.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(wt_gate, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_files = collect_pred_structures(pred_root)
    if max_predictions is not None:
        pred_files = pred_files[:max_predictions]

    rows: list[dict[str, Any]] = []
    for pred_path in pred_files:
        pred_name = pred_path.stem.replace("_model", "")
        backbone_stem = infer_backbone_stem(pred_name)
        row: dict[str, Any] = {
            "pred_name": pred_name,
            "pred_path": str(pred_path),
            "backbone_stem": backbone_stem,
            "motif_eval_enabled": motif_eval_enabled,
        }
        try:
            backbone_pdb, backbone_fasta = find_backbone_files(backbone_root, backbone_stem)
            if backbone_pdb is None or backbone_fasta is None:
                raise FileNotFoundError(f"Backbone PDB/FASTA not found for {backbone_stem}")
            fixed_motif = read_first_sequence(backbone_fasta)
            global_motif_indices = set(extract_fixed_motif_indices(fixed_motif))

            row["backbone_path"] = str(backbone_pdb)
            row["backbone_fixed_motif"] = fixed_motif
            row["motif_residue_count"] = len(global_motif_indices)

            structure_metrics = evaluate_structure(str(backbone_pdb), str(pred_path), usalign_bin=usalign_bin)
            row["backbone_af3_tm_score"] = structure_metrics.get("tm_score")
            row["backbone_af3_rmsd"] = structure_metrics.get("rmsd")

            bb_coords, _ = extract_c1_prime_coords_ordered(str(backbone_pdb))
            pred_coords, _ = extract_c1_prime_coords_ordered(str(pred_path))
            n = min(len(bb_coords), len(pred_coords))
            row["backbone_length"] = len(bb_coords)
            row["pred_length"] = len(pred_coords)
            aligned_pred, _ = kabsch_align(bb_coords[:n], pred_coords[:n])
            lddt, per_res_lddt = compute_lddt(bb_coords[:n], aligned_pred)
            row["backbone_af3_lddt"] = lddt
            row["backbone_af3_lddt_per_residue"] = per_res_lddt

            row.update(compute_compactness_metrics(pred_coords, global_motif_indices))
            row["motif_scaffold_contact_count"] = compute_motif_scaffold_contacts(
                pred_coords,
                global_motif_indices,
            )

            for idx, spec in enumerate(motif_specs, start=1):
                if not spec.get("motif_ref_pdb"):
                    continue
                name = spec.get("name") or f"motif_{idx}"
                spec_enabled = is_motif_eval_enabled(spec, idx)
                row[f"{name}_motif_eval_enabled"] = spec_enabled
                if spec_enabled:
                    name = spec.get("name") or f"motif_{idx}"
                    indices = motif_indices_from_spec(spec, fallback_fixed_motif=fixed_motif)
                    motif_metrics = eval_motif_alignment(
                        pred_path=str(pred_path),
                        motif_ref_pdb=spec["motif_ref_pdb"],
                        motif_indices=indices,
                        usalign_bin=usalign_bin,
                    )
                    for key, value in motif_metrics.items():
                        row[f"{name}_{key}"] = value
                else:
                    row[f"{name}_motif_eval_skipped_reason"] = wt_gate.get("reason")

        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
        rows.append(row)

    write_csv(output_csv, rows)
    ok = sum(1 for row in rows if "error" not in row)
    return {
        "success": True,
        "pred_dir": str(pred_root),
        "backbone_dir": str(backbone_root),
        "output_dir": str(out_root),
        "output_csv": str(output_csv),
        "wt_motif_gate_json": str(gate_path),
        "total_predictions": len(pred_files),
        "evaluated": ok,
        "failed": len(rows) - ok,
        "motif_eval_enabled": motif_eval_enabled,
        "motif_eval_enabled_by_motif": motif_eval_enabled_by_motif,
        "wt_motif_gate": wt_gate,
        "required_inputs": [
            "pred_dir: AF3 prediction directory containing *_model.cif/.pdb files",
            "backbone_dir: filtered RDiff_str backbone directory with matching .pdb and .fasta files",
            "output_dir: directory where results.csv and wt_motif_gate.json are written",
            "motif_specs: optional list of motif refs and indices/fixed motifs for WT gate and design motif evaluation",
        ],
    }


@register_tool("scaffolding_structure_eval")
class ScaffoldingStructureEvalTool(BaseTool):
    description = (
        "Evaluate AF3 predictions from RNA motif-scaffolding pipelines. "
        "First optionally checks whether AF3 reproduces the wild-type motif; "
        "if the WT motif check fails and enable_motif_eval='auto', designed motif "
        "RMSD/TM metrics are skipped. Always reports backbone-AF3 self-consistency, "
        "C1'-lDDT, compactness, and motif-scaffold contact metrics."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pred_dir": {
                "type": "string",
                "description": "AF3 prediction directory containing subdirectories with *_model.cif or .pdb files.",
            },
            "backbone_dir": {
                "type": "string",
                "description": "Filtered RDiff_str backbone directory containing matching .pdb and .fasta files.",
            },
            "output_dir": {
                "type": "string",
                "description": "Output directory for results.csv and wt_motif_gate.json.",
            },
            "motif_specs": {
                "type": "array",
                "description": (
                    "Optional motif specifications. Each item may include name, motif_ref_pdb, "
                    "fixed_motif or motif_indices for designed sequences, and wt_pred_path plus "
                    "wt_fixed_motif or wt_motif_indices for WT motif precheck."
                ),
            },
            "enable_motif_eval": {
                "type": "string",
                "default": "auto",
                "description": "auto/true/false. auto enables motif eval separately for each motif only if its WT motif check passes.",
            },
            "wt_motif_rmsd_threshold": {
                "type": "number",
                "default": 4.0,
                "description": "WT motif check passes only if motif RMSD is <= this threshold.",
            },
            "wt_motif_tm_threshold": {
                "type": "number",
                "default": 0.45,
                "description": "WT motif check passes only if motif TM-score is >= this threshold when available.",
            },
            "usalign_bin": {
                "type": "string",
                "description": "Optional USalign binary path.",
            },
            "max_predictions": {
                "type": "integer",
                "description": "Optional limit for smoke tests.",
            },
        },
        "required": ["pred_dir", "backbone_dir", "output_dir"],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        try:
            result = scaffolding_structure_eval(
                pred_dir=args["pred_dir"],
                backbone_dir=args["backbone_dir"],
                output_dir=args["output_dir"],
                motif_specs=args.get("motif_specs") or [],
                enable_motif_eval=args.get("enable_motif_eval", "auto"),
                wt_motif_rmsd_threshold=args.get("wt_motif_rmsd_threshold", 4.0),
                wt_motif_tm_threshold=args.get("wt_motif_tm_threshold", 0.45),
                usalign_bin=args.get("usalign_bin", DEFAULT_USALIGN_BIN),
                max_predictions=args.get("max_predictions"),
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
