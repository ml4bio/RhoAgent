"""
Convert USalign output into a readable summary plus motif-based natural language notes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import subprocess

from .eval_utils import DEFAULT_USALIGN_BIN


def run_usalign(
    gt_path: str,
    pred_path: str,
    usalign_bin: str = DEFAULT_USALIGN_BIN,
    return_output: bool = False,
) -> Tuple[Optional[float], Optional[float]] | Tuple[Optional[float], Optional[float], List[str]]:
    """
    Run USalign without outfmt and parse TM-score/RMSD from default text output.
    If return_output=True, also return stdout lines.
    """
    try:
        completed = subprocess.run(
            [usalign_bin, gt_path, pred_path, '-TMscore', '2', '-atom', " C1'", '-ter', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        lines = completed.stdout.strip().splitlines()

        tm = None
        rmsd = None

        # 典型行示例: "User-specified initial alignment: TM/Lali/rmsd = 0.28023,   67, 10.611"
        for line in lines:
            if 'TM/Lali/rmsd' in line:
                try:
                    after_eq = line.split('=', 1)[1]
                    tokens = after_eq.replace(',', ' ').split()
                    numbers = []
                    for tok in tokens:
                        try:
                            numbers.append(float(tok))
                        except ValueError:
                            continue
                    if len(numbers) >= 3:
                        tm = numbers[0]
                        rmsd = numbers[2]
                except Exception:
                    tm = None
                break

        if return_output:
            return tm, rmsd, lines
        return tm, rmsd
    except subprocess.CalledProcessError:
        if return_output:
            return None, None, []
        return None, None


def _extract_alignment_block(lines: List[str]) -> List[str]:
    """
    Grab the alignment marker line: the 4th line from the end in default output.
    If missing, return empty list.
    """

    block = lines[-4].strip()
    return block


def _describe_motif_alignment(alignment_block: List[str], motif_template: str) -> List[str]:
    """
    Using a motif template (e.g., G----UUCACGU...), describe alignment per base.
    alignment_block is the marker line; ':' means <5Å, '.' means aligned but >5Å/mismatch.
    """
    if len(alignment_block) < 3:
        return []

    markers = alignment_block
    descriptions: List[str] = []

    for idx, base in enumerate(motif_template):
        if base in ("-", " "):
            continue
        marker = markers[idx] if idx < len(markers) else " "
        if marker == ":":
            status = "Aligned <5Å"
        elif marker == ".":
            status = "Aligned but >5Å or mismatch"
        else:
            status = "Unaligned"
        descriptions.append(f"Pos {idx+1} motif base {base}: {status}")

    return descriptions


def usalign_to_nl(
    gt_path: str,
    pred_path: str,
    motif_template: str = "",
    usalign_bin: str = DEFAULT_USALIGN_BIN,
) -> Dict[str, Any]:
    """
    Run USalign (no outfmt) and extract:
    - TM score / RMSD
    - Marker line near the tail of output
    - Motif-based alignment description (only positions with bases, '-' ignored)
    """
    tm, rmsd, lines = run_usalign(gt_path, pred_path, usalign_bin=usalign_bin, return_output=True)
    alignment_block = _extract_alignment_block(lines)
    motif_desc = _describe_motif_alignment(alignment_block, motif_template) if motif_template else []

    # 在描述末尾补充总体 TM-score 与 RMSD
    if tm is not None or rmsd is not None:
        tm_str = f"{tm:.5f}" if tm is not None else "N/A"
        rmsd_str = f"{rmsd:.3f}" if rmsd is not None else "N/A"
        motif_desc.append(f"The Overall Alignment Score: TM-score={tm_str}, RMSD={rmsd_str}")
    

    motif_desc = ','.join(motif_desc)

    return {
        "gt_path": gt_path,
        "pred_path": pred_path,
        "tm_score": tm,
        "rmsd": rmsd,
        "alignment_block": alignment_block,  # 典型为三行：seq1 / 标记 / seq2
        "motif_template": motif_template,
        "motif_alignment_desc": motif_desc,
        "raw_output_lines": lines,
    }

__all__ = ["usalign_to_nl"]

