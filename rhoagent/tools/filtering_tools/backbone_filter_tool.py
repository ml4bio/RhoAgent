import os
import json
import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from rhoagent.tools.base import BaseTool, register_tool

@register_tool('backbone_filter')
class BackboneFilterTool(BaseTool):
    description = 'Post-processing tool to remove structures that do not meet secondary structure constraints.'
    parameters = {
        'type': 'object',
        'properties': {
            'input_dir': {
                'type': 'string',
                'description': 'Input directory containing PDB files'
            },
            'dot_bracket': {
                'type': 'string',
                'description': 'Target secondary structure dot-bracket string'
            },
            'clash_th': {
                'type': 'number',
                'description': 'Distance threshold for clash detection',
                'default': 3.0
            }
        },
        'required': ['input_dir', 'dot_bracket']
    }

    def call(self, params: str, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        input_dir = args['input_dir']
        dot_bracket = args['dot_bracket']
        clash_th = args.get('clash_th', 3.0)

        try:
            # Perform distance evaluation and save to CSV
        output_csv = self.distance_eval_dir(input_dir, dot_bracket, clash_th)
        
            # Perform filtering and get statistics
            filter_stats = self.filter_structures(output_csv, input_dir)
        
            # Combine results
            result = {
                "success": True,
                "input_dir": input_dir,
                "evaluation_csv": output_csv,
                "dot_bracket": dot_bracket,
                "clash_threshold": clash_th,
                **filter_stats
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Error during backbone filtering: {str(e)}",
                "input_dir": input_dir
            }, ensure_ascii=False)

    def filter_structures(self, csv_path, input_dir):
        """
        Filter structures based on metrics.
        Remove structures if:
        - correct_ratio < 0.8 OR
        - norm_dist_ratio_9.5_12 < 0.8 OR
        - res_dist_ratio_4_6 < 0.6
        
        Returns:
            dict with filtering statistics
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Identify rows to remove
            # Condition: correct_ratio < 0.8 | norm_dist_ratio_9.5_12 < 0.8 | res_dist_ratio_4_6 < 0.6
            
            # Ensure columns exist (handle potential empty DF or missing cols if script changed)
            required_cols = ['pdb_name', 'correct_ratio', 'norm_dist_ratio_9.5_12', 'res_dist_ratio_4_6']
            if not all(col in df.columns for col in required_cols):
                return {
                    "success": False,
                    "error": "Missing required columns in CSV",
                    "missing_columns": [col for col in required_cols if col not in df.columns]
                }

            total_count = len(df)

            # Define the condition
            condition = (
                (df['correct_ratio'] < 0.8) | 
                (df['norm_dist_ratio_9.5_12'] < 0.8) | 
                (df['res_dist_ratio_4_6'] < 0.6)
            )
            
            to_remove = df[condition]['pdb_name'].tolist()
            removed_count = 0
            failed_removals = []
            
            for pdb_file in to_remove:
                file_path = os.path.join(input_dir, pdb_file)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        print(f"Removed {pdb_file} (failed filter criteria)")
                    except OSError as e:
                        failed_removals.append({"file": pdb_file, "error": str(e)})
                        print(f"Error removing {pdb_file}: {e}")
            
            remaining_count = total_count - removed_count
            filter_rate = removed_count / total_count if total_count > 0 else 0.0
            
            return {
                "success": True,
                "total_count": total_count,
                "removed_count": removed_count,
                "remaining_count": remaining_count,
                "filter_rate": filter_rate,
                "filter_criteria": {
                    "correct_ratio_threshold": 0.8,
                    "norm_dist_ratio_9.5_12_threshold": 0.8,
                    "res_dist_ratio_4_6_threshold": 0.6,
                    "description": "Remove structures if correct_ratio < 0.8 OR norm_dist_ratio_9.5_12 < 0.8 OR res_dist_ratio_4_6 < 0.6"
                },
                "failed_removals": failed_removals if failed_removals else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error during filtering: {str(e)}"
            }
            
    def extract_3atom_from_pdb(self, pdb_parser, pdb_path, total_length=None):
        """
        Extract 3 atoms (C4', C1', N9 for purines; C4', C1', N1 for pyrimidines) from a PDB file.
        """
        structure = pdb_parser.get_structure('', pdb_path)
        struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}
        chain = list(struct_chains.values())[0]

        if total_length is None:
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
                continue
            
            atom_names = None
            if res.get_resname() in ['A', 'G']:
                atom_names = ["C4'", "C1'", "N9"]
                seq[res_id] = res.get_resname()
            elif res.get_resname() in ['C', 'U']:
                atom_names = ["C4'", "C1'", "N1"]
                seq[res_id] = res.get_resname()

            res_pos = np.zeros((3, 3))
            if atom_names is not None:
                for i, atom_name in enumerate(atom_names):
                    if atom_name in res:
                        res_pos[i] = res[atom_name].coord

            if np.any((np.sum(res_pos, axis=-1) == 0)):
                missing_mask[res_id] = 1.
            else:
                missing_mask[res_id] = 0.
                pos[res_id] = res_pos

        return pos, missing_mask, ''.join(seq)

    def db_to_contact_map(self, dot_bracket):
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
                    # Warning could be logged
                    pass
        
        return contact_map

    def distance_eval(self, pdb_path, dot_bracket_str, clash_th=3.):
        input_pos, _, _ = self.extract_3atom_from_pdb(PDBParser(PERMISSIVE=1, QUIET=1), pdb_path)

        tmp = input_pos[:, 1].copy()
        res_dist = np.linalg.norm(tmp[1:] - tmp[:-1], axis=1)

        pair_dist = np.linalg.norm(tmp[:, None] - tmp[None, :], axis=2)
        num_clash = np.sum(pair_dist < clash_th) - pair_dist.shape[0]

        contact_map = self.db_to_contact_map(dot_bracket_str)
        
        if contact_map.shape[0] != input_pos.shape[0]:
            min_len = min(contact_map.shape[0], input_pos.shape[0])
            contact_map = contact_map[:min_len, :min_len]
            input_pos = input_pos[:min_len]

        edge_index_norm = np.nonzero(contact_map)
        if len(edge_index_norm[0]) == 0:
            norm_dist_mean = 0.0
            correct_ratio = 0.0
            norm_dist_ratio_95_12 = 0.0
        else:
            norm_dist = np.linalg.norm(input_pos[edge_index_norm[0], 1] - input_pos[edge_index_norm[1], 1], axis=1)
            norm_dist_mean = np.mean(norm_dist)

            correct_dist = np.sum(np.abs(norm_dist - 10.5) < 0.5)
            correct_ratio = correct_dist / len(norm_dist)

            norm_dist_count_95_12 = np.sum((norm_dist > 9.5) & (norm_dist < 12))
            norm_dist_ratio_95_12 = norm_dist_count_95_12 / len(norm_dist)
        
        if len(res_dist) > 0:
            res_dist_count_4_6 = np.sum((res_dist > 4) & (res_dist < 6))
            res_dist_ratio_4_6 = res_dist_count_4_6 / len(res_dist)
        else:
            res_dist_ratio_4_6 = 0.0
            
        return res_dist.mean(), num_clash, norm_dist_mean, correct_ratio, norm_dist_ratio_95_12, res_dist_ratio_4_6

    def distance_eval_dir(self, dir_path, dot_bracket_str, clash_th=3.0):
        pdb_files = [
            os.path.relpath(os.path.join(root, f), dir_path)
            for root, _, files in os.walk(dir_path)
            for f in files if f.endswith('.pdb')
        ]
        # print('pdb_files: ', pdb_files)
        # print('dir_path: ', dir_path)
        data = {
            'pdb_name': [],
            'norm_dist_mean': [],
            'res_dist_mean': [],
            'num_clash': [],
            'correct_ratio': [],
            'norm_dist_ratio_9.5_12': [],
            'res_dist_ratio_4_6': []
        }

        for pdb_file in pdb_files:
            pdb_path = os.path.join(dir_path, pdb_file)
            try:
                metrics = self.distance_eval(pdb_path, dot_bracket_str, clash_th)
                res_dist_mean, num_clash, norm_dist_mean, correct_ratio, norm_dist_ratio, res_dist_ratio = metrics
                
                data['pdb_name'].append(pdb_file)
                data['norm_dist_mean'].append(norm_dist_mean)
                data['res_dist_mean'].append(res_dist_mean)
                data['num_clash'].append(num_clash)
                data['correct_ratio'].append(correct_ratio)
                data['norm_dist_ratio_9.5_12'].append(norm_dist_ratio)
                data['res_dist_ratio_4_6'].append(res_dist_ratio)
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")

        # print('data: ', data)
        df = pd.DataFrame(data)
        df = df.sort_values('pdb_name')
        df_path = os.path.join(dir_path, 'structure_eval.csv')
        df.to_csv(df_path, index=False)
        return df_path
