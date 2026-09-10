"""
Utility functions for RNA structure evaluation.
"""

import os
import sys
import glob
import json
import csv
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from .pdb_utils import extract_rna_sequence_to_fasta, _get_parser, _residue_to_base
from ...struc_tools.utils.rnapolis_utils import structure_to_dotbracket
from Bio.PDB import MMCIFParser, PDBParser

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: create a dummy tqdm that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Try to import concurrent.futures for parallel processing
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    HAS_CONCURRENT = True
except ImportError:
    HAS_CONCURRENT = False

# Try to import pairwise2 from Bio
try:
    from Bio import pairwise2
    HAS_PAIRWISE2 = True
except ImportError:
    HAS_PAIRWISE2 = False

# Directory containing this file.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_USALIGN_BIN = os.path.join(_CURRENT_DIR, 'USalign')





def seq_rec_rate(seq1,seq2):
    """
    Calculate the sequence recovery rate
    """
    assert len(seq1) == len(seq2) # make sure the length of two sequences are the same
    beat = 0
    for index,i in enumerate(seq1):
        if i == seq2[index]:
            beat += 1
    return beat/len(seq1)


def gc_content(seq):
    """
    Calculate the GC content
    """
    return (seq.count('G') + seq.count('C')) / len(seq)


def seq_sim(seq0: str, seq1: str) -> Optional[float]:
    """
    Calculate sequence similarity using local alignment (similar to pairwise2.align.localxx)
    Returns the identity percentage of the best local alignment.
    """
    if not HAS_PAIRWISE2:
        # Fallback: simple character-by-character comparison if lengths match
        if len(seq0) == len(seq1):
            matches = sum(1 for a, b in zip(seq0, seq1) if a == b)
            return matches / len(seq0) if len(seq0) > 0 else None
        return None
    
    try:
        alignments = pairwise2.align.localxx(seq0, seq1)
        if len(alignments) > 0:
            c = 0
            for i in range(len(alignments[0].seqA)):
                if alignments[0].seqA[i] == alignments[0].seqB[i]:
                    c += 1
            score = c / len(alignments[0].seqA) if len(alignments[0].seqA) > 0 else None
            return score
        else:
            return None
    except Exception:
        return None


def calculate_f1_score_dotbracket(pred_ss: str, target_ss: str, fixed_motif_indices: Optional[set] = None) -> Optional[float]:
    """
    Calculate F1-score between two dot-bracket structures.
    F1-score is calculated based on base pairs: TP / (TP + 0.5 * (FP + FN))
    
    Args:
        pred_ss: Predicted secondary structure in dot-bracket notation
        target_ss: Target secondary structure in dot-bracket notation
        fixed_motif_indices: Optional set of indices to consider (for functional core F1-score)
                            If provided, only pairs where both positions are in this set are considered
    """
    def extract_pairs(dotbracket: str) -> set:
        """Extract base pairs from dot-bracket notation"""
        pairs = set()
        stack = []
        for i, char in enumerate(dotbracket):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    pairs.add((j, i))
        return pairs
    
    pred_pairs = extract_pairs(pred_ss)
    target_pairs = extract_pairs(target_ss)
    
    # If fixed_motif_indices is provided, filter pairs to only include those within the motif
    if fixed_motif_indices is not None:
        pred_pairs = {pair for pair in pred_pairs if pair[0] in fixed_motif_indices and pair[1] in fixed_motif_indices}
        target_pairs = {pair for pair in target_pairs if pair[0] in fixed_motif_indices and pair[1] in fixed_motif_indices}
    
    tp = len(pred_pairs & target_pairs)  # True positives
    fp = len(pred_pairs - target_pairs)  # False positives
    fn = len(target_pairs - pred_pairs)  # False negatives
    
    if tp + fp + fn == 0:
        return 1.0 if len(pred_ss) == len(target_ss) and pred_ss == target_ss else 0.0
    
    f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) > 0 else 0.0
    return f1


def calculate_symbol_f1_dotbracket(
    pred_ss: str,
    target_ss: str,
    fixed_motif_indices: Optional[set] = None
) -> Optional[float]:
    """
    Calculate an F1-like score using position-wise dot-bracket symbol matches.
    This compares symbols at each position (for example '.', '(', ')', '[',
    ']', '{', '}') and does not explicitly parse or compare base pairs.

    If fixed_motif_indices is provided, only those positions are evaluated.
    """
    if len(pred_ss) != len(target_ss):
        return None

    if fixed_motif_indices is None:
        indices = range(len(pred_ss))
    else:
        indices = [i for i in fixed_motif_indices if i < len(pred_ss)]
        if not indices:
            return None

    tp = 0  # Positions with matching symbols.
    mismatches = 0  # Positions with different symbols.

    for i in indices:
        if pred_ss[i] == target_ss[i]:
            tp += 1
        else:
            mismatches += 1

    if tp + mismatches == 0:
        return None

    # F1-like form analogous to pair-level F1: TP / (TP + 0.5 * (FP + FN)).
    f1 = tp / (tp + 0.5 * mismatches)
    return f1


def extract_fixed_motif_indices(fixed_motif: str) -> set:
    """
    Extract indices of fixed motif positions from a motif string.
    Example: "------AAU----UUUA-----" -> {6, 7, 8, 12, 13, 14, 15}
    
    Args:
        fixed_motif: Motif string where non-dash characters represent fixed positions
        
    Returns:
        Set of indices where fixed motif bases are located
    """
    indices = set()
    for i, char in enumerate(fixed_motif):
        if char != '-':
            indices.add(i)
    return indices


def extract_pairs_from_dotbracket(dotbracket: str, sequence: str) -> List[Tuple[int, int, str, str]]:
    """
    Extract base pairs from dot-bracket notation.
    Returns list of (i, j, base_i, base_j) tuples.
    """
    pairs = []
    stack = []
    for i, char in enumerate(dotbracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                if i < len(sequence) and j < len(sequence):
                    pairs.append((j, i, sequence[j], sequence[i]))
    return pairs


def extract_pseudoknot_pairs(dotbracket: str, sequence: str) -> List[Tuple[int, int, str, str]]:
    """
    Extract pseudoknot pairs (represented by [] and {} brackets) from dot-bracket notation.
    Returns list of (i, j, base_i, base_j) tuples.
    """
    pairs = []
    stack_square = []  # for []
    stack_curly = []   # for {}
    for i, char in enumerate(dotbracket):
        if char == '[':
            stack_square.append(i)
        elif char == ']':
            if stack_square:
                j = stack_square.pop()
                if i < len(sequence) and j < len(sequence):
                    pairs.append((j, i, sequence[j], sequence[i]))
        elif char == '{':
            stack_curly.append(i)
        elif char == '}':
            if stack_curly:
                j = stack_curly.pop()
                if i < len(sequence) and j < len(sequence):
                    pairs.append((j, i, sequence[j], sequence[i]))
    return pairs


def calculate_canonical_pair_ratio(pairs: List[Tuple[int, int, str, str]]) -> Dict[str, float]:
    """
    Calculate the ratio of AU, GC, GU pairs among all pairs.
    Returns dict with 'AU', 'GC', 'GU' ratios.
    """
    if not pairs:
        return {'AU': 0.0, 'GC': 0.0, 'GU': 0.0, 'total_pairs': 0}
    
    canon_pairs = {'AU': 0, 'UA': 0, 'GC': 0, 'CG': 0, 'GU': 0, 'UG': 0}
    total = len(pairs)
    
    for _, _, base_i, base_j in pairs:
        pair_type = base_i + base_j
        if pair_type in canon_pairs:
            canon_pairs[pair_type] += 1
    
    # Combine AU/UA, GC/CG, GU/UG
    au_count = canon_pairs['AU'] + canon_pairs['UA']
    gc_count = canon_pairs['GC'] + canon_pairs['CG']
    gu_count = canon_pairs['GU'] + canon_pairs['UG']
    
    return {
        'AU': au_count / total if total > 0 else 0.0,
        'GC': gc_count / total if total > 0 else 0.0,
        'GU': gu_count / total if total > 0 else 0.0,
        'total_pairs': total
    }


def calculate_pseudoknot_recall(pred_ss: str, target_ss: str) -> Optional[float]:
    """
    Calculate recall over pseudoknot positions.
    This is the fraction of target pseudoknot positions ([ ] { }) that are also
    annotated as pseudoknot positions in pred_ss.
    
    recall = TP / (TP + FN)
    where:
        TP: positions that are pseudoknot symbols in both target and prediction
        FN: target pseudoknot positions that are not pseudoknot symbols in prediction
    """
    if len(pred_ss) != len(target_ss):
        return None

    def _pk_positions(s: str) -> set:
        return {i for i, ch in enumerate(s) if ch in "[]{}"}

    target_pk = _pk_positions(target_ss)
    if not target_pk:
        # Recall is undefined when the target has no pseudoknot positions.
        return None

    pred_pk = _pk_positions(pred_ss)

    tp = len(target_pk & pred_pk)
    fn = len(target_pk - pred_pk)

    if tp + fn == 0:
        return None

    return tp / (tp + fn)


def extract_c1_prime_coords(structure_path: str, sequence: str, fixed_motif_indices: set) -> Optional[np.ndarray]:
    """
    Extract C1' atom coordinates for positions specified by fixed_motif_indices.

    Args:
        structure_path: Path to PDB or CIF file
        sequence: RNA sequence (0-based indexing)
        fixed_motif_indices: Set of 0-based indices in the sequence to extract

    Returns:
        Numpy array of shape (n, 3) containing C1' coordinates, or None if extraction fails
    """
    try:
        parser = _get_parser(Path(structure_path).suffix)
        structure = parser.get_structure("X", structure_path)

        coords = []
        indices_found = []

        # Get the first chain (assuming single chain RNA)
        chain = next(structure.get_chains(), None)
        if chain is None:
            return None

        # Iterate through residues in order
        residue_list = list(chain.get_residues())

        # Match residues to sequence positions
        seq_idx = 0
        for residue in residue_list:
            if seq_idx >= len(sequence):
                break

            # Check if this is an RNA residue
            base = _residue_to_base(residue)
            if base is None:
                continue

            # Check if the base matches the sequence
            if base != sequence[seq_idx]:
                # Base mismatch - skip or continue?
                # For now, continue to next residue
                seq_idx += 1
                continue

            if seq_idx in fixed_motif_indices:
                # Try to get C1' atom
                try:
                    c1_prime = residue["C1'"]
                    coords.append(c1_prime.get_coord())
                    indices_found.append(seq_idx)
                except KeyError:
                    # Try alternative names
                    try:
                        c1_prime = residue["C1*"]
                        coords.append(c1_prime.get_coord())
                        indices_found.append(seq_idx)
                    except KeyError:
                        # Skip if C1' not found
                        pass

            seq_idx += 1

        if len(coords) == 0:
            return None

        # Check if we got all required indices
        if len(indices_found) != len(fixed_motif_indices):
            # Partial match - still return what we have
            pass

        return np.array(coords)
    except Exception as e:
        print(f"Error extracting coordinates from {structure_path}: {e}")
        return None


def kabsch_align(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Align two sets of coordinates using Kabsch algorithm and calculate RMSD.

    Args:
        coords1: Reference coordinates (N x 3)
        coords2: Coordinates to align (N x 3)

    Returns:
        Tuple of (aligned_coords2, rmsd)
    """
    if len(coords1) != len(coords2):
        raise ValueError("Coordinate sets must have the same length")

    if len(coords1) == 0:
        raise ValueError("Coordinate sets cannot be empty")

    # Center both coordinate sets
    center1 = coords1.mean(axis=0)
    center2 = coords2.mean(axis=0)

    coords1_centered = coords1 - center1
    coords2_centered = coords2 - center2

    # Compute covariance matrix
    H = coords2_centered.T @ coords1_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation and translation
    coords2_aligned = (coords2_centered @ R.T) + center1

    # Calculate RMSD
    diff = coords1 - coords2_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return coords2_aligned, rmsd


def calculate_functional_core_rmsd(
    gt_path: str,
    pred_path: str,
    gt_seq: str,
    pred_seq: str,
    fixed_motif: str
) -> Optional[float]:
    """
    Calculate RMSD for functional core (fixed_motif positions) between two structures.

    Args:
        gt_path: Path to ground truth structure file
        pred_path: Path to predicted structure file
        gt_seq: Ground truth sequence
        pred_seq: Predicted sequence
        fixed_motif: Fixed motif string (e.g., "-----------UUACAA---------------G-GUA--------------------UAA-------------")

    Returns:
        RMSD value for functional core positions, or None if calculation fails
    """
    try:
        # Extract fixed motif indices (0-based) based on gt_seq
        fixed_motif_indices_gt = extract_fixed_motif_indices(fixed_motif)

        if len(fixed_motif_indices_gt) == 0:
            return None

        # Extract the motif sequence from gt_seq
        motif_seq_gt = ''.join([gt_seq[i] if i < len(gt_seq) else '' for i in sorted(fixed_motif_indices_gt)])

        # Find corresponding positions in pred_seq using sequence alignment
        if HAS_PAIRWISE2:
            # Use BioPython pairwise alignment
            alignments = pairwise2.align.globalxx(gt_seq, pred_seq)
            if not alignments:
                return None

            # Get the best alignment
            alignment = alignments[0]
            gt_aligned = alignment.seqA
            pred_aligned = alignment.seqB

            # Map gt_seq indices to pred_seq indices through alignment
            fixed_motif_indices_pred = set()
            gt_to_pred_map = {}
            gt_pos = 0
            pred_pos = 0

            for i in range(len(gt_aligned)):
                if gt_aligned[i] != '-':
                    if pred_aligned[i] != '-':
                        gt_to_pred_map[gt_pos] = pred_pos
                        pred_pos += 1
                    gt_pos += 1
                elif pred_aligned[i] != '-':
                    pred_pos += 1

            # Get corresponding indices in pred_seq
            for gt_idx in fixed_motif_indices_gt:
                if gt_idx in gt_to_pred_map:
                    pred_idx = gt_to_pred_map[gt_idx]
                    fixed_motif_indices_pred.add(pred_idx)
        else:
            # Fallback: assume sequences are aligned (same length and positions match)
            if len(gt_seq) == len(pred_seq):
                fixed_motif_indices_pred = fixed_motif_indices_gt
            else:
                # Try to find motif sequence in pred_seq
                fixed_motif_indices_pred = set()
                for gt_idx in fixed_motif_indices_gt:
                    if gt_idx < len(pred_seq):
                        fixed_motif_indices_pred.add(gt_idx)

        if len(fixed_motif_indices_pred) == 0:
            return None

        # Extract coordinates for fixed motif positions
        gt_coords = extract_c1_prime_coords(gt_path, gt_seq, fixed_motif_indices_gt)
        pred_coords = extract_c1_prime_coords(pred_path, pred_seq, fixed_motif_indices_pred)

        if gt_coords is None or pred_coords is None:
            return None

        # Check if we have matching number of coordinates
        if len(gt_coords) != len(pred_coords):
            # Use the minimum length
            min_len = min(len(gt_coords), len(pred_coords))
            gt_coords = gt_coords[:min_len]
            pred_coords = pred_coords[:min_len]

        if len(gt_coords) == 0:
            return None

        # Align using Kabsch algorithm and calculate RMSD
        try:
            _, rmsd = kabsch_align(gt_coords, pred_coords)
            return float(rmsd)
        except Exception as e:
            print(f"Error in Kabsch alignment: {e}")
            return None
    except Exception as e:
        print(f"Error calculating functional core RMSD: {e}")
        return None


def run_usalign(gt_path: str, pred_path: str, usalign_bin: str = DEFAULT_USALIGN_BIN) -> Tuple[Optional[float], Optional[float]]:
    # Run USalign with outfmt 2 for easy parsing
    try:
        # print('usalign_bin: ', usalign_bin)
        completed = subprocess.run(
            [usalign_bin, gt_path, pred_path, '-TMscore','2','-atom',' C1\'','-ter','1','-outfmt', '2'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        lines = completed.stdout.strip().splitlines()
        # Typical outfmt 2: line 2 contains TM-score, line 3 may have RMSD
        # print('lines: ', lines)
        tm = None
        rmsd = None

        # print('lines: ', lines)
        try:
            tm = float(lines[1].split()[2])
            rmsd = float(lines[1].split()[4])
        except Exception:
            print('Error: Failed to parse TM-score and RMSD from USalign output')
            tm = None

        return tm, rmsd
    except subprocess.CalledProcessError as e:
        # print('Error: Failed to run USalign', e)
        return None, None


def run_usalign_with_alignment_info(
    gt_path: str, 
    pred_path: str, 
    usalign_bin: str = DEFAULT_USALIGN_BIN
) -> Dict[str, Any]:
    """
    Run USalign and parse alignment visualization to get detailed alignment information.
    
    Returns:
        Dictionary containing:
        - tm_score: TM-score value
        - rmsd: RMSD value
        - alignment_description: Natural language description of well-aligned residues
        - aligned_positions: List of 1-based positions that are well-aligned (distance < 5.0 Å)
    """
    try:
        completed = subprocess.run(
            [usalign_bin, gt_path, pred_path, '-TMscore', '2', '-atom', ' C1\'', '-ter', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        output = completed.stdout
        
        # Parse TM-score and RMSD
        tm_score = None
        rmsd = None
        alignment_line = None
        seq1_line = None
        seq2_line = None
        
        lines = output.strip().splitlines()
        for i, line in enumerate(lines):
            # Extract TM-score and RMSD
            if 'TM-score=' in line and 'normalized by length of Structure_1' in line:
                try:
                    parts = line.split('TM-score=')
                    if len(parts) > 1:
                        tm_score = float(parts[1].split()[0])
                except:
                    pass
            if 'RMSD=' in line and 'Seq_ID=' in line:
                try:
                    parts = line.split('RMSD=')
                    if len(parts) > 1:
                        rmsd = float(parts[1].split(',')[0].strip())
                except:
                    pass
            
            # Find alignment visualization section
            if 'denotes residue pairs of d < 5.0' in line.lower():
                # The next few lines should contain the alignment visualization
                if i + 3 < len(lines):
                    seq1_line = lines[i + 1].strip()
                    alignment_line = lines[i + 2].strip()
                    seq2_line = lines[i + 3].strip()
        
        # Parse alignment positions
        aligned_positions = []
        alignment_description = ""
        
        if alignment_line:
            # Find positions with ':' (well-aligned, distance < 5.0 Å)
            for idx, char in enumerate(alignment_line):
                if char == ':':
                    # Position is 1-based
                    aligned_positions.append(idx + 1)
            
            # Generate natural language description
            if aligned_positions:
                # Group consecutive positions
                groups = []
                start = aligned_positions[0]
                end = aligned_positions[0]
                
                for i in range(1, len(aligned_positions)):
                    if aligned_positions[i] == end + 1:
                        end = aligned_positions[i]
                    else:
                        if start == end:
                            groups.append(f"position {start}")
                        else:
                            groups.append(f"positions {start}-{end}")
                        start = aligned_positions[i]
                        end = aligned_positions[i]
                
                # Add last group
                if start == end:
                    groups.append(f"position {start}")
                else:
                    groups.append(f"positions {start}-{end}")
                
                # Generate description
                if len(groups) == 1:
                    alignment_description = f"Residues at {groups[0]} are well-aligned (distance < 5.0 Å)."
                elif len(groups) <= 5:
                    alignment_description = f"Residues at {', '.join(groups)} are well-aligned (distance < 5.0 Å)."
                else:
                    # Too many groups, summarize
                    alignment_description = f"Residues at {len(aligned_positions)} positions are well-aligned (distance < 5.0 Å), including {', '.join(groups[:3])} and {len(groups) - 3} more regions."
            else:
                alignment_description = "No residues are well-aligned (all distances >= 5.0 Å)."
        
        return {
            'tm_score': tm_score,
            'rmsd': rmsd,
            'alignment_description': alignment_description,
            'aligned_positions': aligned_positions,
            'num_aligned': len(aligned_positions) if aligned_positions else 0,
            'alignment_line': alignment_line,
            'seq1_line': seq1_line,
            'seq2_line': seq2_line
        }
    except subprocess.CalledProcessError as e:
        return {
            'tm_score': None,
            'rmsd': None,
            'alignment_description': f"Error running USalign: {e}",
            'aligned_positions': [],
            'num_aligned': 0,
            'alignment_line': None,
            'seq1_line': None,
            'seq2_line': None
        }
    except Exception as e:
        return {
            'tm_score': None,
            'rmsd': None,
            'alignment_description': f"Error parsing USalign output: {e}",
            'aligned_positions': [],
            'num_aligned': 0,
            'alignment_line': None,
            'seq1_line': None,
            'seq2_line': None
        }


def evaluate_structure(
    gt_path: str,
    pred_path: str,
    usalign_bin: str = DEFAULT_USALIGN_BIN,
) -> Dict[str, Any]:
    tm, rmsd = run_usalign(gt_path, pred_path, usalign_bin=usalign_bin)

    return {
        'gt_path': gt_path,
        'pred_path': pred_path,
        'tm_score': None if tm is None else float(tm),
        'rmsd': None if rmsd is None else float(rmsd)
    }

def evaluate_sequence(
    gt_seq: str,
    pred_seq: str,
    
) -> Dict[str, Any]:
    seq_rec_rate_value = seq_rec_rate(gt_seq, pred_seq)
    gc_content_value = gc_content(pred_seq)
    return {
        'gt_seq': gt_seq,
        'pred_seq': pred_seq,
        'seq_rec_rate': seq_rec_rate_value,
        'gc_content': gc_content_value
    }

def _evaluate_single_structure(
    gt_file: str,
    gt_name: str,
    pred_file: str,
    gt_seq: str,
    usalign_bin: str,
    target_ss: Optional[str],
    fixed_motif: Optional[str],
) -> Dict[str, Any]:
    """
    Evaluate a single structure pair (internal function for parallel processing).
    
    Returns:
        Dictionary with evaluation metrics or error information
    """
    pred_name = Path(pred_file).stem
    try:
        # Extract prediction sequence.
        pred_seq_str = extract_rna_sequence_to_fasta(pred_file)[0]
        
        # Evaluate structure.
        struc_metrics = evaluate_structure(
            gt_path=gt_file,
            pred_path=pred_file,
            usalign_bin=usalign_bin,
        )

        if pred_seq_str and gt_seq:
            seq_metrics = evaluate_sequence(
                gt_seq=gt_seq,
                pred_seq=pred_seq_str,
            )
            # Calculate sequence similarity with seq_sim.
            seq_similarity = seq_sim(gt_seq, pred_seq_str)
            seq_metrics['seq_similarity'] = seq_similarity
        else:
            seq_metrics = {'seq_rec_rate': None, 'seq_similarity': None, 'error': 'Failed to extract sequence or no GT sequence'}

        # Evaluate secondary structure when target_ss is provided.
        ss_metrics = {}
        if target_ss:
            try:
                # Extract dot-bracket from the predicted structure.
                pred_dotbracket_result = structure_to_dotbracket(
                    pred_file,
                    output_path=None
                )
                pred_dotbracket_data = json.loads(pred_dotbracket_result)
                
                if pred_dotbracket_data.get('success') and pred_dotbracket_data.get('dot_bracket'):
                    pred_ss = pred_dotbracket_data['dot_bracket']
                    pred_seq_for_ss = pred_dotbracket_data.get('sequence', pred_seq_str or '')
                    
                    # Calculate F1-score.
                    f1_score = calculate_f1_score_dotbracket(pred_ss, target_ss)
                    ss_metrics['ss_f1score'] = f1_score
                    ss_metrics['pred_ss'] = pred_ss
                    ss_metrics['target_ss'] = target_ss
                    
                    # Calculate functional_core_f1score when fixed_motif is provided.
                    if fixed_motif:
                        try:
                            fixed_motif_indices = extract_fixed_motif_indices(fixed_motif)
                            # Define functional_core_f1score using position-wise
                            # dot-bracket symbol matches in the fixed motif region.
                            functional_core_f1score = calculate_symbol_f1_dotbracket(
                                pred_ss, target_ss, fixed_motif_indices=fixed_motif_indices
                            )
                            ss_metrics['functional_core_f1score'] = functional_core_f1score
                            ss_metrics['fixed_motif'] = fixed_motif
                        except Exception as e:
                            ss_metrics['functional_core_f1score'] = None
                            ss_metrics['fixed_motif_error'] = str(e)
                    
                    # Calculate the total AU/GC/GU canonical-pair ratio in target_ss.
                    target_pairs = extract_pairs_from_dotbracket(target_ss, gt_seq)
                    target_pair_ratios = calculate_canonical_pair_ratio(target_pairs)
                    # Sum AU+GC+GU ratios.
                    ss_pairs_ratio = target_pair_ratios['AU'] + target_pair_ratios['GC'] + target_pair_ratios['GU']
                    ss_metrics['ss_pairs_ratio'] = ss_pairs_ratio
                    
                    # Calculate canonical AU/GC/GU ratio for pseudoknot pairs in
                    # the designed/predicted structure, not in target_ss. Using
                    # target_ss here would make this metric constant for all
                    # designs with the same target.
                    designed_pk_pairs = extract_pseudoknot_pairs(pred_ss, pred_seq_for_ss)
                    if designed_pk_pairs:
                        designed_pk_ratios = calculate_canonical_pair_ratio(designed_pk_pairs)
                        designed_pk_canonical_ratio = (
                            designed_pk_ratios["AU"]
                            + designed_pk_ratios["GC"]
                            + designed_pk_ratios["GU"]
                        )
                        ss_metrics["designed_pseudoknot_canonical_pairs_ratio"] = (
                            designed_pk_canonical_ratio
                        )
                    else:
                        ss_metrics["designed_pseudoknot_canonical_pairs_ratio"] = 0.0

                    # Pseudoknot-position recall: target pseudoknot positions that
                    # are also pseudoknot positions in pred_ss.
                    pseudoknot_recall = calculate_pseudoknot_recall(pred_ss, target_ss)
                    ss_metrics['pseudoknot_recall'] = pseudoknot_recall
                else:
                    ss_metrics['error'] = 'Failed to extract dot-bracket from prediction structure'
            except Exception as e:
                ss_metrics['error'] = f'Failed to evaluate secondary structure: {str(e)}'

        # Merge all metrics.
        metrics = {
            "gt_name": gt_name,
            "pred_name": pred_name,
            **struc_metrics,
            **seq_metrics,
            **ss_metrics
        }
        return metrics
    
    except Exception as exc:
        return {"gt_file": gt_file, "pred_file": pred_file, "error": str(exc)}


def evaluate_batch(
    gt_struc_dir: str,
    gt_seq: str,
    pred_struc_dir: str,
    usalign_bin: str = DEFAULT_USALIGN_BIN,
    pred_seq_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    target_ss: Optional[str] = None,
    fixed_motif: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Batch-evaluate all predicted structures under pred_dir.
    
    Args:
        gt_struc_dir: Ground truth structure directory
        gt_seq: Ground truth sequence
        pred_struc_dir: Predicted structure directory
        usalign_bin: Path to USalign executable
        pred_seq_dir: Optional predicted sequence directory
        output_dir: Optional output directory for results
        target_ss: Optional target secondary structure
        fixed_motif: Optional fixed motif string
        max_workers: Maximum number of parallel workers (None = auto-detect CPU count, 1 = sequential)
    """

    # Find all GT files
    gt_files = []
    for ext in ['*.pdb', '*.cif']:
        gt_files.extend(glob.glob(os.path.join(gt_struc_dir, ext)))
        gt_files.extend(glob.glob(os.path.join(gt_struc_dir, "**", ext), recursive=True))
    
    gt_files = sorted(list(set(gt_files)))
    
    if not gt_files:
        return {"gt_struc_dir": gt_struc_dir, "num_structures": 0, "results": [], "error": "no GT structures found"}

    results: List[Dict[str, Any]] = []
    

    # Pre-compute all GT-prediction pairs for progress bar
    gt_pred_pairs = []
    for gt_file in gt_files:
        gt_name = Path(gt_file).stem
        # Find corresponding prediction file
        # Recursively find all .cif/.pdb files containing gt_name in pred_struc_dir
        pred_candidates = []
        for ext in ['*.pdb', '*.cif']:
             pred_candidates.extend(glob.glob(os.path.join(pred_struc_dir, "**", f"*{gt_name}*{ext}"), recursive=True))
        pred_candidates = sorted(list(set(pred_candidates)))
        
        if not pred_candidates:
            gt_pred_pairs.append((gt_file, gt_name, None))  # No match
        else:
            for pred_file in pred_candidates:
                gt_pred_pairs.append((gt_file, gt_name, pred_file))
    
    print(f"Found {len(gt_files)} GT structures, {len(gt_pred_pairs)} structure pairs to evaluate")
    
    # Filter out None pairs (no match) and handle them separately
    valid_pairs = [(gt_file, gt_name, pred_file) for gt_file, gt_name, pred_file in gt_pred_pairs if pred_file is not None]
    no_match_results = [{"gt_name": gt_name, "error": "No matching prediction found"} 
                        for gt_file, gt_name, pred_file in gt_pred_pairs if pred_file is None]
    
    results: List[Dict[str, Any]] = []
    
    # Use parallel processing if available and more than one task
    if HAS_CONCURRENT and len(valid_pairs) > 1 and (max_workers is None or max_workers > 1):
        # Determine number of workers
        if max_workers is None:
            import multiprocessing
            max_workers = min(multiprocessing.cpu_count(), len(valid_pairs), 8)  # Cap at 8 to avoid too many threads
        
        print(f"Using {max_workers} parallel workers for evaluation")
        
        # Use tqdm for progress bar (thread-safe)
        pbar = tqdm(total=len(valid_pairs), desc="Evaluating structures", unit="structure")
        
        # Submit all tasks to thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(
                    _evaluate_single_structure,
                    gt_file, gt_name, pred_file, gt_seq, usalign_bin, target_ss, fixed_motif
                ): (gt_file, gt_name, pred_file)
                for gt_file, gt_name, pred_file in valid_pairs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_pair):
                gt_file, gt_name, pred_file = future_to_pair[future]
                try:
                    result = future.result()
                    results.append(result)
                    pred_name = Path(pred_file).stem if pred_file else "N/A"
                    pbar.update(1)
                    pbar.set_postfix({"completed": len(results), "current": pred_name[:30]})
                except Exception as exc:
                    results.append({"gt_file": gt_file, "pred_file": pred_file, "error": str(exc)})
                    pbar.update(1)
        
        pbar.close()
    else:
        # Fallback to sequential processing
        if not HAS_CONCURRENT:
            print("Warning: concurrent.futures not available, using sequential processing")
        else:
            print("Using sequential processing (only 1 structure pair)")
        
        # Use tqdm for progress bar
        pbar = tqdm(total=len(valid_pairs), desc="Evaluating structures", unit="structure")
        
        for gt_file, gt_name, pred_file in valid_pairs:
            result = _evaluate_single_structure(
                gt_file, gt_name, pred_file, gt_seq, usalign_bin, target_ss, fixed_motif
            )
            results.append(result)
            pred_name = Path(pred_file).stem
            pbar.update(1)
            pbar.set_postfix({"completed": len(results), "current": pred_name[:30]})
        
        pbar.close()
    
    # Add no-match results
    results.extend(no_match_results)

    # Save results to CSV file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, "results.csv")
        
        if results:
            # Flatten nested dictionaries and convert complex types to strings
            flattened_results = []
            for result in results:
                flattened = {}
                for key, value in result.items():
                    if value is None:
                        flattened[key] = ''
                    elif isinstance(value, (dict, list)):
                        # Convert complex types to JSON string
                        flattened[key] = json.dumps(value, ensure_ascii=False) if value else ''
                    elif isinstance(value, (int, float, str, bool)):
                        flattened[key] = value
                    else:
                        # Convert other types to string
                        flattened[key] = str(value)
                flattened_results.append(flattened)
            
            # Get all unique fieldnames from all results
            fieldnames = set()
            for result in flattened_results:
                fieldnames.update(result.keys())
            fieldnames = sorted(list(fieldnames))
            
            # Write to CSV
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_results)
            
            results_file = csv_file
        else:
            # Create empty CSV with headers if no results
            results_file = csv_file
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['error'])
                writer.writerow(['No results to save'])
    else:
        results_file = None

    return {
        "gt_struc_dir": gt_struc_dir, 
        "pred_struc_dir": pred_struc_dir, 
        "num_structures": len(results), 
        "results": results, 
        "results_file": results_file
    }
