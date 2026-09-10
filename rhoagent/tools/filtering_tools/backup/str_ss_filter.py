import pdb
import numpy as np
import pandas as pd
import os
from Bio.PDB.PDBParser import PDBParser
# from analysis import utils as au

def extract_3atom_from_pdb(pdb_parser, pdb_path, total_length=None):
    """
    Extract 3 atoms (C4', C1', N9 for purines; C4', C1', N1 for pyrimidines) from a PDB file.

    Return:
        pos: np.array [N, 3, 3], N is the number of residues, 3 is the number of atoms (C4', C1', N9/N1)
        missing mask: np.array [N], True if the residue is missing, False otherwise
        seq: str, sequence of the RNA
    """

    structure = pdb_parser.get_structure('', pdb_path)
    struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}
    chain = list(struct_chains.values())[0]

    if total_length is None:
        # Determine length from residues
        # Assuming 1-based indexing in PDB and we want to cover up to the max index
        res_ids = [res.id[1] for res in chain]
        if not res_ids:
            total_length = 0
        else:
            total_length = max(res_ids)
    
    seq = ['-' for _ in range(total_length)]
    pos = np.zeros((total_length, 3, 3))
    missing_mask = np.ones(total_length)

    for res in chain:
        res_id = res.id[1] - 1
        if res_id >= total_length:
             continue # Skip residues outside the determined length (though shouldn't happen if auto-detected)
        if res.get_resname() in ['A', 'G']:
            atom_names = ["C4'", "C1'", "N9"]
            seq[res_id] = res.get_resname()
        elif res.get_resname() in ['C', 'U']:
            atom_names = ["C4'", "C1'", "N1"]
            seq[res_id] = res.get_resname()
        else:
            atom_names = None

        # extract atom coordinates
        res_pos = np.zeros((3, 3))
        if atom_names is not None:
            for i, atom_name in enumerate(atom_names):
                if atom_name in res:
                    res_pos[i] = res[atom_name].coord

        # check if all three atoms are present
        if np.any((np.sum(res_pos, axis=-1) == 0)):
            missing_mask[res_id] = 1.
        else:
            missing_mask[res_id] = 0.
            pos[res_id] = res_pos

    return pos, missing_mask, ''.join(seq)



def db_to_contact_map(dot_bracket):
    """
    Convert dot-bracket string (supporting (), [], {}) to contact map.
    """
    length = len(dot_bracket)
    contact_map = np.zeros((length, length), dtype=int)
    
    stacks = {
        '(': [],
        '[': [],
        '{': [],
    }
    
    pairs = {
        ')': '(',
        ']': '[',
        '}': '{',
    }
    
    for i, char in enumerate(dot_bracket):
        if char in stacks:
            stacks[char].append(i)
        elif char in pairs:
            open_char = pairs[char]
            if stacks[open_char]:
                j = stacks[open_char].pop()
                contact_map[i, j] = 1
                contact_map[j, i] = 1
            else:
                print(f"Warning: Unmatched closing bracket '{char}' at index {i}")
    
    for char, stack in stacks.items():
        if stack:
             print(f"Warning: Unmatched opening brackets '{char}' at indices {stack}")
             
    return contact_map

def distance_eval(pdb_path, dot_bracket_str, clash_th=3.):
    # read pdb file
    input_pos, _, _ = extract_3atom_from_pdb(PDBParser(PERMISSIVE=1, QUIET=1), pdb_path)

    # get the distance between next residues
    tmp = input_pos[:, 1].copy()
    res_dist = np.linalg.norm(tmp[1:] - tmp[:-1], axis=1)

    # pair wise distance
    pair_dist = np.linalg.norm(tmp[:, None] - tmp[None, :], axis=2)
    # get the num of dist < 2
    num_clash = np.sum(pair_dist < clash_th) - pair_dist.shape[0]

    # parse dot bracket to contact map
    contact_map = db_to_contact_map(dot_bracket_str)
    
    # Check if contact map size matches pdb size
    if contact_map.shape[0] != input_pos.shape[0]:
        # If they don't match, we might want to align or warn. 
        # For now, assume the PDB parsing determines the correct length and if DB is diff, slice or error.
        # Often PDB might be longer or shorter if missing residues, but extract_3atom fills missing with 0.
        # However, extract_3atom now auto-detects length from PDB. 
        # If dot_bracket is the ground truth, we should probably use its length?
        # But input_pos is from PDB.
        # Let's check min length.
        min_len = min(contact_map.shape[0], input_pos.shape[0])
        contact_map = contact_map[:min_len, :min_len]
        input_pos = input_pos[:min_len]
        # This is a naive fix, ideally they should match.

    # get normal base pairs
    edge_index_norm = np.nonzero(contact_map)
    if len(edge_index_norm[0]) == 0:
        norm_dist_mean = 0.0
        correct_ratio = 0.0
        norm_dist_ratio_95_12 = 0.0
    else:
        norm_dist = np.linalg.norm(input_pos[edge_index_norm[0], 1] - input_pos[edge_index_norm[1], 1], axis=1)
        norm_dist_mean = np.mean(norm_dist)

        # if the distance for base pair is 10.5 +- 0.5 is the correct one, calculate the correct ratio
        correct_dist = np.sum(np.abs(norm_dist - 10.5) < 0.5)
        correct_ratio = correct_dist / len(norm_dist)

        # norm_dist in [9.5, 12] ratio
        norm_dist_count_95_12 = np.sum((norm_dist > 9.5) & (norm_dist < 12))
        norm_dist_ratio_95_12 = norm_dist_count_95_12 / len(norm_dist)
    
    # res_dist in [4, 6] ratio
    if len(res_dist) > 0:
        res_dist_count_4_6 = np.sum((res_dist > 4) & (res_dist < 6))
        res_dist_ratio_4_6 = res_dist_count_4_6 / len(res_dist)
    else:
        res_dist_ratio_4_6 = 0.0
        
    return res_dist.mean(), num_clash, norm_dist_mean, correct_ratio, norm_dist_ratio_95_12, res_dist_ratio_4_6


def distance_eval_dir(dir_path, dot_bracket_str):
    # print(f'PDB path: {dir_path}, Dot Bracket: {dot_bracket_str}')
    # get all pdb files in the directory
    pdb_files = [f for f in os.listdir(dir_path) if f.endswith('.pdb')]
    pdb_names = []
    norm_dist_mean_list = []
    res_dist_list = []
    num_clash_list = []
    correct_ratio_list = []
    norm_dist_ratio_list = []
    res_dist_ratio_list = []

    for pdb_file in pdb_files:
        pdb_path = os.path.join(dir_path, pdb_file)
        try:
            res_dist_mean, num_clash, norm_dist_mean, correct_ratio, norm_dist_ratio, res_dist_ratio = distance_eval(pdb_path, dot_bracket_str)
            pdb_names.append(pdb_file)
            norm_dist_mean_list.append(norm_dist_mean)
            res_dist_list.append(res_dist_mean)
            num_clash_list.append(num_clash)
            correct_ratio_list.append(correct_ratio)
            norm_dist_ratio_list.append(norm_dist_ratio)
            res_dist_ratio_list.append(res_dist_ratio)
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

    # write into csv files with pdb_name, norm_dist_mean, pk_dist_mean as head
    df = pd.DataFrame({
        'pdb_name': pdb_names, 
        'norm_dist_mean': norm_dist_mean_list,
        'res_dist_mean': res_dist_list, 
        'num_clash': num_clash_list, 
        'correct_ratio': correct_ratio_list,
        'norm_dist_ratio_9.5_12': norm_dist_ratio_list,
        'res_dist_ratio_4_6': res_dist_ratio_list
    })

    # sort by pdb_name
    df = df.sort_values('pdb_name')
    df_path = os.path.join(dir_path, 'structure_eval.csv')
    df.to_csv(df_path, index=False)

if __name__ == '__main__':
    # Example dot bracket string, replace with actual input
    example_db = "((((...((.(((....)))....))...))))" 
    distance_eval_dir('/workspace/hedongchen/RhoDesign/tmmp', example_db)
