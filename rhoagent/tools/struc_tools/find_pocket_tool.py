"""
Find pocket tool:

Based on a given RNA 3D structure (CIF or PDB),
starting from user-provided cut site bases (seed residues, default positions 16 and 17),
iteratively expand based on C4' atom distance, return the corresponding "dash-base sequence" for the entire chain (unselected positions are '-', selected positions are base letters).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

from rhoagent.tools.base import BaseTool, register_tool


def _calculate_distance(coord1, coord2) -> float:
    """Calculate Euclidean distance between two coordinate points"""
    return float(np.linalg.norm(np.array(coord1) - np.array(coord2)))


def _get_c4_prime_coord(residue: Residue):
    """Get C4' atom coordinate of residue"""
    for atom in residue:
        if atom.get_id() == "C4'":
            return atom.get_coord()
    return None


def _find_residue_by_seq_id(chain: Chain, seq_id: int) -> Residue | None:
    """Find residue by sequence ID"""
    for residue in chain:
        if residue.get_id()[1] == seq_id:
            return residue
    return None


def _load_first_chain(structure_path: str) -> Chain:
    """Load structure file and return first chain of first model"""
    if not os.path.exists(structure_path):
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    ext = os.path.splitext(structure_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    elif ext in [".pdb", ".ent"]:
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported structure format: {ext}, expected .cif/.mmcif/.pdb/.ent")

    structure = parser.get_structure("rna_structure", structure_path)
    # Get first model and first chain
    model = structure[0]
    chain = None
    for c in model:
        chain = c
        break

    if chain is None:
        raise ValueError("No chain found in structure")
    return chain


def _chain_to_sequence(chain: Chain) -> Tuple[List[int], str]:
    """Convert chain to (position list, single-letter sequence)"""
    full_sequence: List[Tuple[int, str]] = []
    for residue in chain:
        seq_id = residue.get_id()[1]
        resname = residue.get_resname().strip()
        if resname in ["A", "ADE", "A  "]:
            base = "A"
        elif resname in ["U", "URA", "U  "]:
            base = "U"
        elif resname in ["G", "GUA", "G  "]:
            base = "G"
        elif resname in ["C", "CYT", "C  "]:
            base = "C"
        else:
            base = "?"
        full_sequence.append((seq_id, base))

    full_sequence.sort(key=lambda x: x[0])
    positions = [seq_id for seq_id, _ in full_sequence]
    seq_line = "".join(base for _, base in full_sequence)
    return positions, seq_line


def find_pocket_from_structure(
    structure_path: str,
    target_positions: List[int] | None = None,
    round1_threshold: float = 14.0,
    round2_threshold: float = 9.5,
    exclude_window: int = 5,
    max_rounds: int = 2,
) -> str:
    """
    Core logic: In given structure, starting from specified seed positions, iteratively expand based on C4' distance, return dash-base sequence.

    Return value is a JSON string with fields:
    - success: Whether successful
    - structure_path: Absolute path to structure file
    - pocket_sequence: Bases shown only at selected positions, others are '-'
    - original_sequence: Original full-length base sequence
    - positions: Sequence number list corresponding to each position
    - selected_positions: List of all selected positions
    """
    try:
        chain = _load_first_chain(structure_path)

        # Collect all residues and sort by sequence position
        all_residues = []
        for residue in chain:
            seq_id = residue.get_id()[1]
            all_residues.append((seq_id, residue))
        all_residues.sort(key=lambda x: x[0])

        if len(all_residues) < 1:
            return json.dumps(
                {
                    "success": False,
                    "error": "No residues found in chain",
                    "structure_path": os.path.abspath(structure_path),
                },
                ensure_ascii=False,
            )

        # Default seeds: positions 16 and 17, consistent with original script
        if target_positions is None:
            target_positions = [16, 17]

        # Get C4' coordinates of seed residues
        seed_residues: Dict[int, Dict] = {}
        for pos in target_positions:
            residue = _find_residue_by_seq_id(chain, pos)
            if not residue:
                continue
            c4_coord = _get_c4_prime_coord(residue)
            if c4_coord is None:
                continue
            seed_residues[pos] = {
                "residue": residue,
                "c4_coord": c4_coord,
                "resname": residue.get_resname().strip(),
                "round": 0,
            }

        if not seed_residues:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Could not find valid C4' atoms for seed positions: {target_positions}",
                    "structure_path": os.path.abspath(structure_path),
                },
                ensure_ascii=False,
            )

        # Store all found bases and their information
        all_found_residues: Dict[int, Dict] = {}
        for pos, info in seed_residues.items():
            all_found_residues[pos] = {
                "residue": info["residue"],
                "c4_coord": info["c4_coord"],
                "resname": info["resname"],
                "found_from": {},
                "round": 0,
            }

        # Iterative search
        current_round = 0
        new_residues_found = True
        while new_residues_found and current_round < max_rounds:
            current_round += 1
            new_residues_found = False

            if current_round == 1:
                current_threshold = round1_threshold
            else:
                current_threshold = round2_threshold

            # Bases to search in current round (found in previous round)
            search_from = {
                seq_id: info
                for seq_id, info in all_found_residues.items()
                if info["round"] == current_round - 1
            }
            if not search_from:
                break

            # Collect all positions to exclude (surrounding ±exclude_window bases of all search sources)
            exclude_positions = set()
            for source_seq_id in search_from.keys():
                for offset in range(-exclude_window, exclude_window + 1):
                    exclude_positions.add(source_seq_id + offset)

            # Iterate through all residues, calculate distance with each search source
            for residue in chain:
                seq_id = residue.get_id()[1]

                if seq_id in exclude_positions:
                    continue

                if (
                    seq_id in all_found_residues
                    and all_found_residues[seq_id]["round"] < current_round
                ):
                    continue

                c4_coord = _get_c4_prime_coord(residue)
                if c4_coord is None:
                    continue

                found_from_this_round: Dict[int, float] = {}
                for source_seq_id, source_info in search_from.items():
                    source_coord = source_info["c4_coord"]
                    distance = _calculate_distance(c4_coord, source_coord)
                    if distance < current_threshold:
                        found_from_this_round[source_seq_id] = distance

                if found_from_this_round:
                    if seq_id not in all_found_residues:
                        resname = residue.get_resname().strip()
                        all_found_residues[seq_id] = {
                            "residue": residue,
                            "c4_coord": c4_coord,
                            "resname": resname,
                            "found_from": found_from_this_round,
                            "round": current_round,
                        }
                        new_residues_found = True
                    elif all_found_residues[seq_id]["round"] == current_round:
                        all_found_residues[seq_id]["found_from"].update(
                            found_from_this_round
                        )

        # Generate sequence and dash sequence
        positions, original_sequence = _chain_to_sequence(chain)
        selected_positions = set(all_found_residues.keys())

        pocket_chars: List[str] = []
        for seq_id, base in zip(positions, original_sequence):
            if seq_id in selected_positions:
                pocket_chars.append(base)
            else:
                pocket_chars.append("-")
        pocket_sequence = "".join(pocket_chars)

        result = {
            "success": True,
            "structure_path": os.path.abspath(structure_path),
            "pocket_sequence": pocket_sequence,
            "original_sequence": original_sequence,
            "positions": positions,
            "selected_positions": sorted(selected_positions),
            "round1_threshold": round1_threshold,
            "round2_threshold": round2_threshold,
            "exclude_window": exclude_window,
            "max_rounds": max_rounds,
            "target_positions": target_positions,
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "error": str(e),
                "structure_path": os.path.abspath(structure_path),
            },
            ensure_ascii=False,
        )


@register_tool("find_pocket")
class FindPocketTool(BaseTool):
    """Tool for iteratively finding pockets in RNA 3D structure from given cut sites/seed bases, returning dash-base sequence."""

    description = (
        "Given an RNA structure file (CIF or PDB) and a list of seed base sequence numbers corresponding to cut sites (e.g., [16,17]), "
        "iteratively expand nearby bases from C4' atoms of these seed residues based on distance thresholds, and return a dash-base sequence aligned with the original sequence (unselected positions are '-')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "structure_path": {
                "type": "string",
                "description": "RNA structure file path, supports .cif/.mmcif/.pdb/.ent",
            },
            "seed_positions": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of seed base sequence numbers corresponding to cut sites, e.g., [16,17]; defaults to [16,17] if not provided",
            },
            "round1_threshold": {
                "type": "number",
                "default": 14.0,
                "description": "C4' distance threshold (Å) for first round iteration, default 14.0",
            },
            "round2_threshold": {
                "type": "number",
                "default": 9.5,
                "description": "C4' distance threshold (Å) for second and subsequent round iterations, default 9.5",
            },
            "exclude_window": {
                "type": "integer",
                "default": 5,
                "description": "Number of residues to exclude on both sides of each search source (±window), default 5",
            },
            "max_rounds": {
                "type": "integer",
                "default": 2,
                "description": "Maximum number of iteration rounds (excluding seed round), default 2",
            },
        },
        "required": ["structure_path"],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        return find_pocket_from_structure(
            structure_path=args["structure_path"],
            target_positions=args.get("seed_positions"),
            round1_threshold=args.get("round1_threshold", 14.0),
            round2_threshold=args.get("round2_threshold", 9.5),
            exclude_window=args.get("exclude_window", 5),
            max_rounds=args.get("max_rounds", 2),
        )


