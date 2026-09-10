"""
RNA sequence filtering tool.

Filters FASTA records by GC content, homopolymer runs, duplicate sequences,
repeating subsequences, and optional sequence recovery rate against a reference
sequence. The default sequence recovery rule keeps records with recovery rate
below 0.8, matching the first-round diversity-oriented filtering convention.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator, Optional

from rhoagent.tools.base import BaseTool, register_tool


def parse_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    parts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, normalize_sequence("".join(parts))
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line)
        if header is not None:
            yield header, normalize_sequence("".join(parts))


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.upper().split()).replace("T", "U")


def read_first_fasta_sequence(path: Path) -> str:
    for _, sequence in parse_fasta(path):
        return sequence
    raise ValueError(f"No FASTA records found in {path}")


def gc_content(sequence: str) -> float:
    if not sequence:
        return 0.0
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def sequence_recovery_rate(sequence: str, reference_sequence: str) -> Optional[float]:
    if not sequence or not reference_sequence:
        return None
    if len(sequence) != len(reference_sequence):
        return None
    matches = sum(1 for a, b in zip(sequence, reference_sequence) if a == b)
    return matches / len(sequence)


def has_homopolymer_run(sequence: str, run_len: int) -> bool:
    if run_len <= 1:
        return bool(sequence)
    if not sequence:
        return False
    current = sequence[0]
    count = 1
    for base in sequence[1:]:
        if base == current:
            count += 1
            if count >= run_len:
                return True
        else:
            current = base
            count = 1
    return False


def has_repeating_subsequence(sequence: str, min_repeat_length: int) -> bool:
    """Return True if any subsequence of at least min_repeat_length appears twice."""
    if min_repeat_length <= 0:
        return False
    if len(sequence) < min_repeat_length * 2:
        return False
    for length in range(min_repeat_length, len(sequence) // 2 + 1):
        seen: set[str] = set()
        for start in range(0, len(sequence) - length + 1):
            subseq = sequence[start : start + length]
            if subseq in seen:
                return True
            seen.add(subseq)
    return False


def resolve_reference_sequence(
    reference_sequence: str = "",
    reference_fasta: str = "",
) -> Optional[str]:
    if reference_sequence:
        return normalize_sequence(reference_sequence)
    if reference_fasta:
        return read_first_fasta_sequence(Path(reference_fasta))
    return None


def seq_filtering(
    *,
    input_fasta: str,
    output_fasta: Optional[str] = None,
    metrics_csv: Optional[str] = None,
    reference_sequence: str = "",
    reference_fasta: str = "",
    recovery_rate_threshold: float = 0.8,
    gc_content_min: float = 0.35,
    gc_content_max: float = 0.7,
    homopolymer_run_len: int = 6,
    repeat_subseq_min_len: int = 0,
    remove_duplicates: bool = True,
) -> str:
    input_path = Path(input_fasta)
    if not input_path.is_file():
        return json.dumps(
            {"success": False, "error": f"Input FASTA does not exist: {input_fasta}"},
            ensure_ascii=False,
        )

    output_path = Path(output_fasta) if output_fasta else input_path.with_name(f"{input_path.stem}_filtered.fasta")
    metrics_path = Path(metrics_csv) if metrics_csv else input_path.with_name(f"{input_path.stem}_filter_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        reference = resolve_reference_sequence(reference_sequence, reference_fasta)
    except Exception as exc:
        return json.dumps(
            {"success": False, "error": f"Failed to read reference sequence: {exc}"},
            ensure_ascii=False,
        )

    records = list(parse_fasta(input_path))
    if not records:
        return json.dumps(
            {"success": False, "error": f"No FASTA records found in {input_fasta}"},
            ensure_ascii=False,
        )

    seen_sequences: set[str] = set()
    rows: list[dict[str, object]] = []
    kept_records: list[tuple[str, str]] = []
    failure_counts = {
        "duplicate": 0,
        "gc_content": 0,
        "homopolymer": 0,
        "repeating_subsequence": 0,
        "recovery_rate": 0,
        "recovery_rate_unavailable": 0,
    }

    for header, sequence in records:
        gc = gc_content(sequence)
        recovery = sequence_recovery_rate(sequence, reference) if reference else None
        duplicate = sequence in seen_sequences
        has_homopolymer = has_homopolymer_run(sequence, homopolymer_run_len)
        has_repeat = (
            has_repeating_subsequence(sequence, repeat_subseq_min_len)
            if repeat_subseq_min_len > 0
            else False
        )

        reasons: list[str] = []
        if remove_duplicates and duplicate:
            reasons.append("duplicate")
        if not (gc_content_min <= gc <= gc_content_max):
            reasons.append("gc_content")
        if has_homopolymer:
            reasons.append("homopolymer")
        if has_repeat:
            reasons.append("repeating_subsequence")
        if reference is not None:
            if recovery is None:
                reasons.append("recovery_rate_unavailable")
            elif recovery >= recovery_rate_threshold:
                reasons.append("recovery_rate")

        kept = not reasons
        if kept:
            kept_records.append((header, sequence))
            seen_sequences.add(sequence)
        elif not duplicate:
            # Only mark non-kept unique sequences as seen after judging duplicate status.
            seen_sequences.add(sequence)

        for reason in reasons:
            failure_counts[reason] += 1

        rows.append(
            {
                "name": header,
                "seq_len": len(sequence),
                "gc_content": f"{gc:.6f}",
                "recovery_rate": "" if recovery is None else f"{recovery:.6f}",
                "duplicate": duplicate,
                "has_homopolymer": has_homopolymer,
                "has_repeating_subsequence": has_repeat,
                "kept": kept,
                "filter_reasons": ";".join(reasons),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for header, sequence in kept_records:
            f.write(f">{header}\n{sequence}\n")

    fieldnames = [
        "name",
        "seq_len",
        "gc_content",
        "recovery_rate",
        "duplicate",
        "has_homopolymer",
        "has_repeating_subsequence",
        "kept",
        "filter_reasons",
    ]
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_count = len(records)
    kept_count = len(kept_records)
    return json.dumps(
        {
            "success": True,
            "input_fasta": str(input_path),
            "output_fasta": str(output_path),
            "metrics_csv": str(metrics_path),
            "total_count": total_count,
            "kept_count": kept_count,
            "removed_count": total_count - kept_count,
            "retention_rate": kept_count / total_count if total_count else 0.0,
            "filter_criteria": {
                "recovery_rate_threshold": recovery_rate_threshold,
                "recovery_rule": "keep records with recovery_rate < threshold; skipped if no reference is provided",
                "gc_content_range": [gc_content_min, gc_content_max],
                "homopolymer_run_len": homopolymer_run_len,
                "repeat_subseq_min_len": repeat_subseq_min_len,
                "remove_duplicates": remove_duplicates,
            },
            "reference_provided": reference is not None,
            "failure_counts": failure_counts,
        },
        ensure_ascii=False,
    )


@register_tool("seq_filter")
class SeqFilterTool(BaseTool):
    """Filter RNA FASTA records by sequence-level quality and diversity metrics."""

    description = (
        "Filter RNA sequences in a FASTA file by GC content, homopolymer runs, "
        "duplicate sequences, optional repeating subsequences, and optional sequence "
        "recovery rate against a reference sequence. By default, recovery rate must "
        "be lower than 0.8 when a reference is provided."
    )
    parameters = {
        "type": "object",
        "properties": {
            "input_fasta": {
                "type": "string",
                "description": "Input FASTA file containing designed RNA sequences.",
            },
            "output_fasta": {
                "type": "string",
                "description": "Output filtered FASTA path. Default: <input_stem>_filtered.fasta.",
            },
            "metrics_csv": {
                "type": "string",
                "description": "Output per-sequence metrics CSV path. Default: <input_stem>_filter_metrics.csv.",
            },
            "reference_sequence": {
                "type": "string",
                "description": "Reference RNA sequence for sequence recovery rate calculation.",
            },
            "reference_fasta": {
                "type": "string",
                "description": "Reference FASTA path. The first record is used if reference_sequence is not provided.",
            },
            "recovery_rate_threshold": {
                "type": "number",
                "default": 0.8,
                "description": "Keep only sequences with recovery_rate < threshold when a reference is provided.",
            },
            "gc_content_min": {
                "type": "number",
                "default": 0.35,
                "description": "Minimum allowed GC content.",
            },
            "gc_content_max": {
                "type": "number",
                "default": 0.7,
                "description": "Maximum allowed GC content.",
            },
            "homopolymer_run_len": {
                "type": "integer",
                "default": 6,
                "description": "Reject sequences containing this many or more identical consecutive bases.",
            },
            "repeat_subseq_min_len": {
                "type": "integer",
                "default": 0,
                "description": "Reject repeated subsequences of at least this length. Set 0 to disable.",
            },
            "remove_duplicates": {
                "type": "boolean",
                "default": True,
                "description": "Reject duplicate sequences after the first occurrence.",
            },
        },
        "required": ["input_fasta"],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return seq_filtering(
            input_fasta=args["input_fasta"],
            output_fasta=args.get("output_fasta"),
            metrics_csv=args.get("metrics_csv"),
            reference_sequence=args.get("reference_sequence", ""),
            reference_fasta=args.get("reference_fasta", ""),
            recovery_rate_threshold=args.get("recovery_rate_threshold", 0.8),
            gc_content_min=args.get("gc_content_min", 0.35),
            gc_content_max=args.get("gc_content_max", 0.7),
            homopolymer_run_len=args.get("homopolymer_run_len", 6),
            repeat_subseq_min_len=args.get("repeat_subseq_min_len", 0),
            remove_duplicates=args.get("remove_duplicates", True),
        )
