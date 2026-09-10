"""
First-round filtering tool module.

Two filtering modes are supported:
- standard: ribozyme/general eval results, using seq recovery, GC, seq similarity,
  SS-F1, and TM-score.
- scaffolding: motif-scaffolding eval results, using backbone/AF3 consistency,
  compactness, optional motif metrics, and length-aware selection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from rhoagent.tools.base import BaseTool, register_tool


def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _to_int(value, default: Optional[int] = None) -> Optional[int]:
    parsed = _to_float(value, None)
    if parsed is None:
        return default
    return int(parsed)


def _read_csv(input_path: Path) -> tuple[list[dict], list[str]]:
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(output_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = sorted({key for row in rows for key in row})
    else:
        extras = sorted({key for row in rows for key in row} - set(fieldnames))
        fieldnames = fieldnames + extras
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _detect_filter_mode(rows: list[dict], requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    columns = set(rows[0].keys()) if rows else set()
    if {"backbone_af3_tm_score", "backbone_af3_lddt"} & columns:
        return "scaffolding"
    return "standard"


def first_round_filtering(
    input_csv: str,
    output_csv: Optional[str] = None,
    filter_mode: str = "auto",
    seq_rec_rate_threshold: float = 0.5,
    gc_content_min: float = 0.3,
    gc_content_max: float = 0.7,
    seq_similarity_threshold: float = 0.7,
    ss_f1_threshold: float = 0.9,
    top_n: int = 100,
    scaffold_top_n_per_length: int = 10,
    scaffold_min_backbone_tm: float = 0.0,
    scaffold_min_backbone_lddt: float = 0.0,
    scaffold_max_isolated_fraction: float = 1.0,
    scaffold_max_motif_scaffold_contacts: Optional[int] = None,
) -> str:
    """
    Perform first-round filtering on evaluation results.
    
    Args:
        input_csv: Input evaluation results CSV file path
        output_csv: Output filtered CSV file path, auto-generated if None
        seq_rec_rate_threshold: Sequence recovery rate threshold (keep records below this value), default 0.5
        gc_content_min: Minimum GC content, default 0.3
        gc_content_max: Maximum GC content, default 0.7
        seq_similarity_threshold: Sequence similarity threshold (keep records below this value), default 0.7
        ss_f1_threshold: If ss_f1score exists, keep records with ss_f1score >= this value.
        top_n: Number of top records to keep after filtering and sorting (default 100)
    
    Returns:
        JSON-formatted result summary
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        return json.dumps({
            "success": False,
            "error": f"Input file does not exist: {input_csv}",
        }, ensure_ascii=False)
    
    rows, fieldnames = _read_csv(input_path)
    if not rows:
        return json.dumps({
            "success": False,
            "error": "CSV file is empty",
        }, ensure_ascii=False)

    mode = _detect_filter_mode(rows, filter_mode)
    if mode not in {"standard", "scaffolding"}:
        return json.dumps({
            "success": False,
            "error": f"Unknown filter_mode: {filter_mode}. Use auto, standard, or scaffolding.",
        }, ensure_ascii=False)

    if output_csv is None:
        input_dir = input_path.parent
        output_path = input_dir / f"{input_path.stem}_filtered.csv"
    else:
        output_path = Path(output_csv)

    if mode == "scaffolding":
        return scaffolding_first_round_filtering(
            rows=rows,
            fieldnames=fieldnames,
            input_path=input_path,
            output_path=output_path,
            top_n=top_n,
            top_n_per_length=scaffold_top_n_per_length,
            min_backbone_tm=scaffold_min_backbone_tm,
            min_backbone_lddt=scaffold_min_backbone_lddt,
            max_isolated_fraction=scaffold_max_isolated_fraction,
            max_motif_scaffold_contacts=scaffold_max_motif_scaffold_contacts,
        )
    
    # Apply filtering conditions
    filtered_rows = []
    ss_f1_available = any((row.get("ss_f1score") or "").strip() for row in rows)
    for row in rows:
        try:
            seq_rec_rate = _to_float(row.get('seq_rec_rate'), 0.0)
            gc_content = _to_float(row.get('gc_content'), 0.0)
            tm_score = _to_float(row.get('tm_score'), 0.0)
            
            # Filtering conditions:
            # 1. seq_rec_rate < threshold
            # 2. gc_content within [min, max]
            # 3. seq_similarity < threshold (if column exists)
            # 4. ss_f1score >= threshold (if column exists)
            condition1 = seq_rec_rate < seq_rec_rate_threshold
            condition2 = gc_content_min <= gc_content <= gc_content_max
            
            # Check if seq_similarity column exists
            if 'seq_similarity' in row and row['seq_similarity']:
                seq_similarity = float(row['seq_similarity'])
                condition3 = seq_similarity < seq_similarity_threshold
            else:
                # If column doesn't exist, skip this condition (backward compatibility)
                condition3 = True

            if ss_f1_available:
                ss_f1score = _to_float(row.get("ss_f1score"), None)
                condition4 = ss_f1score is not None and ss_f1score >= ss_f1_threshold
            else:
                condition4 = True
            
            if condition1 and condition2 and condition3 and condition4:
                # Keep original row data without modifying any fields
                filtered_rows.append((tm_score, row))
        except (ValueError, KeyError):
            # Skip rows that cannot be parsed
            continue
    
    # Sort by tm_score in descending order (using first element of tuple for sorting)
    filtered_rows.sort(key=lambda x: x[0], reverse=True)
    
    # Extract sorted original row data and keep only top N
    filtered_rows = [row for _, row in filtered_rows[:top_n]]
    
    _write_csv(output_path, filtered_rows, fieldnames)
    
    # Calculate statistics
    total_count = len(rows)
    filtered_count = len(filtered_rows)
    filter_rate = filtered_count / total_count if total_count > 0 else 0.0
    
    # Get best result
    best_result = None
    if filtered_rows:
        try:
            best_result = {
                'pred_name': filtered_rows[0].get('pred_name', ''),
                'tm_score': float(filtered_rows[0].get('tm_score', 0)),
                'seq_rec_rate': float(filtered_rows[0].get('seq_rec_rate', 0)),
                'gc_content': float(filtered_rows[0].get('gc_content', 0)),
            }
        except (ValueError, KeyError):
            pass
    
    return json.dumps({
        "success": True,
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "total_count": total_count,
        "filtered_count": filtered_count,
        "filter_rate": filter_rate,
        "filter_criteria": {
            "seq_rec_rate_threshold": seq_rec_rate_threshold,
            "gc_content_range": [gc_content_min, gc_content_max],
            "seq_similarity_threshold": seq_similarity_threshold,
            "ss_f1_threshold": ss_f1_threshold,
            "ss_f1_applied": ss_f1_available,
            "sort_by": "tm_score (descending)",
            "top_n": top_n,
        },
        "best_result": best_result,
        "filter_mode": "standard",
    }, ensure_ascii=False)


def _scaffold_score(row: dict) -> tuple[float, ...]:
    return (
        _to_float(row.get("backbone_af3_lddt"), -999.0) or -999.0,
        _to_float(row.get("backbone_af3_tm_score"), -999.0) or -999.0,
        -(_to_float(row.get("isolated_scaffold_fraction"), 999.0) or 999.0),
        -(_to_float(row.get("rg_norm"), 999.0) or 999.0),
        -(_to_float(row.get("end_to_end_dist_norm"), 999.0) or 999.0),
        _to_float(row.get("non_motif_long_range_contact_density"), -999.0) or -999.0,
    )


def scaffolding_first_round_filtering(
    *,
    rows: list[dict],
    fieldnames: list[str],
    input_path: Path,
    output_path: Path,
    top_n: int,
    top_n_per_length: int,
    min_backbone_tm: float,
    min_backbone_lddt: float,
    max_isolated_fraction: float,
    max_motif_scaffold_contacts: Optional[int],
) -> str:
    passed: list[dict] = []
    for row in rows:
        if (row.get("error") or "").strip():
            continue
        tm = _to_float(row.get("backbone_af3_tm_score"), None)
        lddt = _to_float(row.get("backbone_af3_lddt"), None)
        isolated = _to_float(row.get("isolated_scaffold_fraction"), 0.0)
        motif_contacts = _to_int(row.get("motif_scaffold_contact_count"), 0)

        if tm is None or tm < min_backbone_tm:
            continue
        if lddt is None or lddt < min_backbone_lddt:
            continue
        if isolated is not None and isolated > max_isolated_fraction:
            continue
        if (
            max_motif_scaffold_contacts is not None
            and motif_contacts is not None
            and motif_contacts > max_motif_scaffold_contacts
        ):
            continue
        passed.append(row)

    grouped: dict[int, list[dict]] = {}
    for row in passed:
        length = _to_int(row.get("backbone_length"), None)
        if length is None:
            length = _to_int(row.get("pred_length"), None)
        if length is None:
            non_motif_len = _to_int(row.get("non_motif_residue_count"), None)
            motif_count = _to_int(row.get("motif_residue_count"), 0) or 0
            length = non_motif_len + motif_count if non_motif_len is not None else None
        if length is None:
            length = _to_int(row.get("seq_len"), -1) or -1
        grouped.setdefault(length, []).append(row)

    selected: list[dict] = []
    length_counts: dict[str, int] = {}
    for length, group_rows in sorted(grouped.items()):
        ranked = sorted(group_rows, key=_scaffold_score, reverse=True)
        keep = ranked[:top_n_per_length]
        selected.extend(keep)
        length_counts[str(length)] = len(keep)

    if len(selected) > top_n:
        # Keep length diversity first: round-robin across length groups.
        by_length = {
            length: sorted(group_rows, key=_scaffold_score, reverse=True)
            for length, group_rows in sorted(grouped.items())
        }
        selected = []
        while len(selected) < top_n and any(by_length.values()):
            for length in sorted(by_length):
                if by_length[length] and len(selected) < top_n:
                    selected.append(by_length[length].pop(0))

    _write_csv(output_path, selected, fieldnames)
    best_result = selected[0] if selected else None
    return json.dumps({
        "success": True,
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "total_count": len(rows),
        "passed_count": len(passed),
        "filtered_count": len(selected),
        "filter_rate": len(selected) / len(rows) if rows else 0.0,
        "filter_mode": "scaffolding",
        "length_aware_selection": True,
        "selected_per_length": length_counts,
        "filter_criteria": {
            "min_backbone_af3_tm_score": min_backbone_tm,
            "min_backbone_af3_lddt": min_backbone_lddt,
            "max_isolated_scaffold_fraction": max_isolated_fraction,
            "max_motif_scaffold_contacts": max_motif_scaffold_contacts,
            "top_n": top_n,
            "top_n_per_length": top_n_per_length,
            "ranking": [
                "higher backbone_af3_lddt",
                "higher backbone_af3_tm_score",
                "lower isolated_scaffold_fraction",
                "lower rg_norm",
                "lower end_to_end_dist_norm",
                "higher non_motif_long_range_contact_density",
            ],
        },
        "best_result": best_result,
    }, ensure_ascii=False)


@register_tool("first_round_filtering")
class FirstRoundFilterTool(BaseTool):
    """First-round filtering tool for standard ribozyme/general evals or scaffolding evals."""

    description = (
        "Perform first-round filtering on evaluation results CSV file. "
        "Mode 'standard' filters by sequence recovery rate, GC content, sequence similarity, ss_f1score>=0.9 when present, "
        "and sorts by TM-score. Mode 'scaffolding' filters by backbone-AF3 consistency and compactness, then keeps candidates "
        "with length-aware stratification so different scaffold lengths are represented."
    )
    parameters = {
        "type": "object",
        "properties": {
            "input_csv": {
                "type": "string",
                "description": "Input evaluation results CSV file path (containing columns like tm_score, seq_rec_rate, gc_content, etc.)",
            },
            "output_csv": {
                "type": "string",
                "description": "Output filtered CSV file path (optional, default generates *_filtered.csv in the same directory as input file)",
            },
            "filter_mode": {
                "type": "string",
                "default": "auto",
                "description": "Filtering mode: auto, standard, or scaffolding. Auto detects scaffolding columns.",
            },
            "seq_rec_rate_threshold": {
                "type": "number",
                "default": 0.5,
                "description": "Sequence recovery rate threshold, only records below this value will be kept (default 0.5)",
            },
            "gc_content_min": {
                "type": "number",
                "default": 0.3,
                "description": "Minimum GC content (default 0.3)",
            },
            "gc_content_max": {
                "type": "number",
                "default": 0.7,
                "description": "Maximum GC content (default 0.7)",
            },
            "seq_similarity_threshold": {
                "type": "number",
                "default": 0.7,
                "description": "Sequence similarity threshold, only records below this value will be kept (default 0.7)",
            },
            "ss_f1_threshold": {
                "type": "number",
                "default": 0.9,
                "description": "If ss_f1score column exists, only records with ss_f1score >= this value are kept in standard mode.",
            },
            "top_n": {
                "type": "integer",
                "default": 100,
                "description": "Number of top records to keep after filtering and sorting by TM-score (default 100)",
            },
            "scaffold_top_n_per_length": {
                "type": "integer",
                "default": 10,
                "description": "Scaffolding mode: keep at most this many candidates per scaffold length before global top_n.",
            },
            "scaffold_min_backbone_tm": {
                "type": "number",
                "default": 0.0,
                "description": "Scaffolding mode: minimum backbone_af3_tm_score.",
            },
            "scaffold_min_backbone_lddt": {
                "type": "number",
                "default": 0.0,
                "description": "Scaffolding mode: minimum backbone_af3_lddt.",
            },
            "scaffold_max_isolated_fraction": {
                "type": "number",
                "default": 1.0,
                "description": "Scaffolding mode: maximum isolated_scaffold_fraction.",
            },
            "scaffold_max_motif_scaffold_contacts": {
                "type": "integer",
                "description": "Scaffolding mode: optional maximum motif_scaffold_contact_count.",
            },
        },
        "required": ["input_csv"],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return first_round_filtering(
            input_csv=args["input_csv"],
            output_csv=args.get("output_csv"),
            filter_mode=args.get("filter_mode", "auto"),
            seq_rec_rate_threshold=args.get("seq_rec_rate_threshold", 0.5),
            gc_content_min=args.get("gc_content_min", 0.3),
            gc_content_max=args.get("gc_content_max", 0.7),
            seq_similarity_threshold=args.get("seq_similarity_threshold", 0.7),
            ss_f1_threshold=args.get("ss_f1_threshold", 0.9),
            top_n=args.get("top_n", 100),
            scaffold_top_n_per_length=args.get("scaffold_top_n_per_length", 10),
            scaffold_min_backbone_tm=args.get("scaffold_min_backbone_tm", 0.0),
            scaffold_min_backbone_lddt=args.get("scaffold_min_backbone_lddt", 0.0),
            scaffold_max_isolated_fraction=args.get("scaffold_max_isolated_fraction", 1.0),
            scaffold_max_motif_scaffold_contacts=args.get("scaffold_max_motif_scaffold_contacts"),
        )

