import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import re
from Bio.PDB import PDBParser
from tqdm import tqdm
import pdb

def read_fasta(fasta_path: str) -> str:
    """Read sequence from FASTA file."""
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    # Skip header line and join sequence lines
    return ''.join(line.strip() for line in lines[1:])

def parse_pdb_name(pdb_name: str) -> dict:
    """
    Parse PDB filename to extract length and contig information.
    Example: 'FRET_translated_31A_scaffold_140_11_A3-16_18_A29-47_18_E3-22_15_E35-47_12_0.pdb' -> 
    {
        'length': 140,
        'contig_info': '11_A3-16_18_A29-47_18_E3-22_15_E35-47_12'
    }
    """
    # Remove .pdb extension if present
    pdb_name = pdb_name.replace('.pdb', '')
    
    # Split by underscore to find the length
    parts = pdb_name.split('_')
    
    # Find the length (it should be a number after 'scaffold')
    length = None
    contig_info = None
    
    for i, part in enumerate(parts):
        if part == 'scaffold' and i + 1 < len(parts):
            # The next part should be the length
            try:
                length = int(parts[i + 1])
                # Everything after the length until the last part (which is usually a number like 0)
                contig_parts = parts[i + 2:-1]  # Exclude the last part (usually 0)
                contig_info = '_'.join(contig_parts)
                break
            except ValueError:
                continue
    
    return {
        'length': length,
        'contig_info': contig_info
    }


def analyze_pdb_clash(pdb_path: str, length: int) -> Dict:
    """
    Analyze PDB structure for clashes between C1' atoms and adjacent residue C1' distances.
    Returns dictionary with clash information and adjacent distance statistics.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', pdb_path)
    
    # Get all three atoms for each residue (original clash calculation)
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # # 'A', 'G': C1', C4', N9; 'U', 'C': C1', C4', N1
                # if residue.get_resname().strip() in ['A', 'G']:
                #     atoms.append(residue['C1\''].get_coord())
                #     atoms.append(residue['C4\''].get_coord())
                #     atoms.append(residue['N9'].get_coord())
                # elif residue.get_resname().strip() in ['U', 'C']:
                #     atoms.append(residue['C1\''].get_coord())
                #     atoms.append(residue['C4\''].get_coord())
                #     atoms.append(residue['N1'].get_coord())
                
                # for generated pdb files, use all atoms
                for atom in residue:
                    atoms.append(atom.get_coord())

    # Convert to numpy array for efficient distance calculation
    atoms_coords = np.array(atoms) # shape: (n*3, 3)
    
    # Calculate pairwise distances using vectorized operations
    atoms_coords_i = atoms_coords[:, np.newaxis, :]  # Shape: (n*3, 1, 3)
    atoms_coords_j = atoms_coords[np.newaxis, :, :]  # Shape: (1, n*3, 3)
    
    # Calculate distances using broadcasting
    distances = np.sqrt(np.sum((atoms_coords_i - atoms_coords_j) ** 2, axis=2))
    
    # Count clashes (distance < 1.1 Å)
    clash_mask = distances < 1.1  # shape: (n*3, n*3)
    np.fill_diagonal(clash_mask, False)  # Exclude self-distances
    
    # Reshape clash mask to group by residues (each residue has 3 atoms)
    clash_mask = clash_mask.reshape(length, 3, length, 3)
    
    # A residue has clash if any of its atoms clash with any atoms of other residues
    residue_clash = np.any(np.any(clash_mask, axis=3), axis=1)  # shape: (n_residues, n_residues)
    np.fill_diagonal(residue_clash, False)  # Exclude self-clashes
    
    # Count number of residues involved in clashes (if a residue clashes with multiple others, count once)
    residues_with_clash = np.any(residue_clash, axis=1)
    clash_ratio = np.sum(residues_with_clash) / length
    
    # Additional analysis: Calculate adjacent residue C1' distances (using res_id)
    c1_atoms = []
    res_ids = []
    rna_resnames = {"A", "U", "G", "C"}  
    for model in structure:
        for chain in model:
            residues = list(chain)
            if not residues:
                continue
            if residues[0].get_resname().strip() not in rna_resnames:
                continue  # skip non-RNA chain
            for residue in residues:
                if residue.get_resname().strip() not in rna_resnames:
                    continue
                if 'C4\'' in residue:
                    c1_atoms.append(residue['C4\''].get_coord())
                    res_ids.append(residue.get_id()[1])
                else:
                    for atom in residue:
                        c1_atoms.append(atom.get_coord())
                        res_ids.append(residue.get_id()[1])
                        break

    # Calculate adjacent residue C1' distances for consecutive res_id
    adjacent_distances = []
    for i in range(len(c1_atoms) - 1):
        if res_ids[i+1] == res_ids[i] + 1:
            dist = np.linalg.norm(c1_atoms[i] - c1_atoms[i + 1])
            adjacent_distances.append(dist)
    assert len(adjacent_distances) == length - 1, f"Length mismatch: {len(adjacent_distances)} != {length - 1}"

    # Count adjacent distances exceeding 8Å
    adjacent_distances = np.array(adjacent_distances)
    long_distance_count = np.sum(adjacent_distances > 8.0)

    return {
        'clash_count': np.sum(residues_with_clash),
        'clash_ratio': clash_ratio,
        'adjacent_long_distance_count': long_distance_count,
    }


def collect_scaffold_pdb(root_dir: str, output_file: str = "scaffold_metadata.csv"):
    """
    Collect all scaffold pdb files from the root directory, save the meta to a csv file.
    
    Args:
        root_dir: Root directory containing subdirectories with PDB and FASTA files
        output_file: Name of the output CSV file (default: "scaffold_metadata.csv")
    """
    data = []
    
    # Get immediate subdirectories
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if len(subdirs) == 0:
        subdirs.append('')

    for subdir_name in subdirs:
        subdir_path = os.path.join(root_dir, subdir_name)

        files = os.listdir(subdir_path)
        
        # Find PDB and FASTA files in current subdirectory
        pdb_files = [f for f in files if f.endswith('.pdb')]
        fasta_files = [f for f in files if f.endswith('.fasta')]
        
        if not pdb_files or not fasta_files:
            continue
        
        # Add each PDB file to the data list
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            pdb_path = os.path.join(subdir_path, pdb_file)

            # Get sequence from FASTA file, _*.pdb -> .fasta
            fasta_path = os.path.join(subdir_path, pdb_file[:-6] + '.fasta')
            sequence = read_fasta(fasta_path)

            # Parse pdb name to get length and contig information
            pdb_info = parse_pdb_name(pdb_file)

            # Analyze PDB structure for clashes
            clash_info = analyze_pdb_clash(pdb_path, pdb_info['length'])
            
            data.append({
                'pdb_path': pdb_path,
                'sequence': sequence,
                'length': pdb_info['length'],
                'contig_info': pdb_info['contig_info'],
                'clash_count': clash_info['clash_count'],
                'clash_ratio': clash_info['clash_ratio'],
                'adjacent_long_distance_count': clash_info['adjacent_long_distance_count']
            })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_path = output_file
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved metadata to {output_path}")
    print(f"Collected {len(df)} PDB files")

if __name__ == "__main__":

    # analyze_pdb_clash('7zj4.pdb', 374)

    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_31A_scaffold',
        output_file='fret_31A_scaffold_metadata.csv')

    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_45A_scaffold',
        output_file='fret_45A_scaffold_metadata.csv')
    
    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_55A_scaffold',
        output_file='fret_55A_scaffold_metadata.csv')
    
    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_65A_scaffold',
        output_file='fret_65A_scaffold_metadata.csv')
    
    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_75A_scaffold',
        output_file='fret_75A_scaffold_metadata.csv')
    
    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_85A_scaffold',
        output_file='fret_85A_scaffold_metadata.csv')
    
    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_110A_scaffold',
        output_file='fret_110A_scaffold_metadata.csv')
    
    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_35A_225_scaffold',
        output_file='fret_35A_225_scaffold_metadata.csv')
    
    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_35A_270_scaffold',
        output_file='fret_35A_270_scaffold_metadata.csv')
    
    collect_scaffold_pdb(
        root_dir='/data/lhong/huanghan/AF3_3DRiboDiffusion/inference_outputs/fret_35A_90_scaffold',
        output_file='fret_35A_90_scaffold_metadata.csv')
    
    