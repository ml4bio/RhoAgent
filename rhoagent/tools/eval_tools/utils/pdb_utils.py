"""
PDB相关实用函数
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from Bio.PDB import MMCIFParser, PDBParser

RNA_RES_MAP: Dict[str, str] = {
    'A': 'A', 'ADE': 'A', 'RA': 'A',
    'U': 'U', 'URA': 'U', 'RU': 'U',
    'G': 'G', 'GUA': 'G', 'RG': 'G',
    'C': 'C', 'CYT': 'C', 'RC': 'C',
}


def _get_parser(ext: str):
    if ext.lower() in {'.cif', '.mmcif'}:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _residue_to_base(residue) -> Optional[str]:
    name = residue.get_resname().strip().upper()
    return RNA_RES_MAP.get(name)


def extract_rna_sequence_to_fasta(pdb_path: str,
                                  fasta_path: str = None,
                                #   output_file: True,
                                  chain_filter: Optional[str] = None) -> Optional[str]:
    """
    从PDB/CIF提取RNA序列，写入FASTA。按链输出，若多个链则多条序列。
    返回写入的路径，若无RNA残基则返回None。
    """
    parser = _get_parser(Path(pdb_path).suffix)
    structure = parser.get_structure("X", pdb_path)
    sequences: List[str] = []
    headers: List[str] = []

    for chain in structure.get_chains():
        if chain_filter and chain.id != chain_filter:
            continue
        seq_chars: List[str] = []
        for residue in chain.get_residues():
            base = _residue_to_base(residue)
            if base:
                seq_chars.append(base)
        if seq_chars:
            sequences.append("".join(seq_chars))
            headers.append(f"{Path(pdb_path).stem}_{chain.id}")

    if not sequences:
        return None


    if fasta_path:
        os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
        with open(fasta_path, 'w', encoding='utf-8') as f:
            for header, seq in zip(headers, sequences):
                f.write(f">{header}\n")
                f.write(seq + "\n")
        return fasta_path
    else:
        return sequences


