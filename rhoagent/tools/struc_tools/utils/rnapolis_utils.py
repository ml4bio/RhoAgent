"""
RNApolis utility wrappers.
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, Any, Optional, List

import rnapolis
from rnapolis.parser import read_3d_structure
from rnapolis.annotator import find_pairs, find_stackings

# Biopython for robust CIF/PDB filtering
try:
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.PDBIO import PDBIO, Select
except Exception:
    MMCIFParser = None
    PDBParser = None
    PDBIO = None
    Select = object  # type: ignore



def _format_pdb_atom_line(
    atom_index: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_num: int,
    x: float,
    y: float,
    z: float,
) -> str:
    if len(atom_name) < 4 and atom_name and atom_name[0].isalpha():
        atom_field = f" {atom_name:<3}"
    else:
        atom_field = f"{atom_name:>4}"
    element = (atom_name[0] if atom_name else ' ').upper()
    return (
        f"ATOM  "
        f"{atom_index:>5d} "
        f"{atom_field}"
        f" "
        f"{res_name:>3} "
        f"{chain_id:1}"
        f"{res_num:>4d}"
        f" "
        f"   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}"
        f"          "
        f"{element:>2}  "
    )


class _RNASelect(Select):  # type: ignore[misc]
    def __init__(self, chain_filter: Optional[str] = None) -> None:
        self.chain_filter = chain_filter
        self.rna_names = {"A", "C", "G", "U", "ADE", "CYT", "GUA", "URA", "A  ", "C  ", "G  ", "U  "}

    def accept_residue(self, residue):  # type: ignore[override]
        if self.chain_filter is not None:
            try:
                ch = residue.get_parent().id
                if ch != self.chain_filter:
                    return 0
            except Exception:
                return 0
        try:
            name = residue.get_resname().strip()
            return 1 if name in self.rna_names else 0
        except Exception:
            return 0

    def accept_atom(self, atom):  # type: ignore[override]
        return 1


def _bio_to_rna_only_pdb(input_path: str, pdb_out: str, is_cif: bool, chain_filter: Optional[str]) -> None:
    if MMCIFParser is None or PDBIO is None:
        raise RuntimeError("Biopython not available for CIF/PDB filtering")
    if is_cif:
        structure = MMCIFParser(QUIET=True).get_structure("X", input_path)
    else:
        structure = PDBParser(QUIET=True).get_structure("X", input_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_out, _RNASelect(chain_filter))


def _pdb_to_rna_only_pdb(pdb_in: str, pdb_out: str, chain_filter: Optional[str]) -> None:
    _bio_to_rna_only_pdb(pdb_in, pdb_out, is_cif=False, chain_filter=chain_filter)


def _generate_dotbracket_from_canon_pairs(structure, canon_pairs) -> str:
    n = len(structure.residues)
    db = ["."] * n
    index = {res: i for i, res in enumerate(structure.residues)}
    for pair in canon_pairs:
        r1 = getattr(pair, 'residue1', None)
        r2 = getattr(pair, 'residue2', None)
        if r1 is None or r2 is None:
            try:
                r1, r2 = pair[0], pair[1]  # type: ignore[index]
            except Exception:
                continue
        i = index.get(r1); j = index.get(r2)
        if i is None or j is None:
            continue
        if 0 <= i < n and 0 <= j < n:
            if i < j:
                db[i] = "("; db[j] = ")"
            else:
                db[i] = ")"; db[j] = "("
    return "".join(db)


def parse_pdb_manually(pdb_path: str, output_path: Optional[str] = None, include_metadata: bool = True) -> str:
    """Manually parse a PDB/CIF file as a fallback method."""
    sequences = []
    dot_brackets = []
    chain_info = []
    
    # Group residues by chain.
    chains = {}
    
    with open(pdb_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Handle ATOM records.
            if line.startswith('ATOM') or (line.startswith('HETATM') and any(base in line for base in ['A', 'U', 'G', 'C'])):
                try:
                    # Parse ATOM records.
                    if line.startswith('ATOM'):
                        # PDB format.
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        chain_id = line[21:22].strip()
                        res_num = int(line[22:26].strip())
                    else:
                        # Simplified CIF parsing.
                        parts = line.split()
                        if len(parts) >= 8:
                            atom_name = parts[3]
                            res_name = parts[5]
                            chain_id = parts[6]
                            try:
                                res_num = int(parts[8])
                            except:
                                res_num = 1
                        else:
                            continue
                    
                    # Check whether this is an RNA residue.
                    if res_name in ['A', 'U', 'G', 'C', 'ADE', 'URA', 'GUA', 'CYT']:
                        if chain_id not in chains:
                            chains[chain_id] = {}
                        
                        if res_num not in chains[chain_id]:
                            chains[chain_id][res_num] = res_name
                            
                except Exception as e:
                    # Skip lines that cannot be parsed.
                    continue
    
    # Generate sequences.
    for chain_id, residues in chains.items():
        # Sort by residue number.
        sorted_residues = sorted(residues.items())
        sequence = ""
        
        for res_num, res_name in sorted_residues:
            # Convert residue names to one-letter base codes.
            if res_name in ['A', 'ADE']:
                sequence += 'A'
            elif res_name in ['U', 'URA']:
                sequence += 'U'
            elif res_name in ['G', 'GUA']:
                sequence += 'G'
            elif res_name in ['C', 'CYT']:
                sequence += 'C'
            else:
                sequence += '?'
        
        if sequence and any(base in sequence for base in 'AUGC'):
            sequences.append(sequence)
            dot_bracket = generate_simple_dotbracket(sequence)
            dot_brackets.append(dot_bracket)
            
            chain_info.append({
                'chain_id': chain_id,
                'sequence_length': len(sequence),
                'num_residues': len(residues),
                'residue_types': dict(residues)
            })
    
    result = {
        'input_file': pdb_path,
        'success': True,
        'sequences': sequences,
        'dot_brackets': dot_brackets,
        'num_chains': len(sequences),
        'chain_info': chain_info,
        'parsing_method': 'manual'
    }
    
    # Save result.
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result['output_file'] = output_path
    
    return json.dumps(result, ensure_ascii=False)


def batch_pdb_to_dotbracket(
    input_dir: str,
    output_dir: str,
    file_pattern: str = "*.pdb"
) -> str:
    """Batch-convert PDB/CIF files to dot-bracket notation."""
    import glob
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Find matching files.
        pattern = os.path.join(input_dir, file_pattern)
        pdb_files = glob.glob(pattern)
        
        results = []
        successful = 0
        failed = 0
        
        for pdb_file in pdb_files:
            filename = os.path.basename(pdb_file)
            name_without_ext = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, f"{name_without_ext}_dotbracket.json")
            
            try:
                # Process one file.
                result_json = structure_to_dotbracket(pdb_file, output_file)
                result = json.loads(result_json)
                
                if result['success']:
                    successful += 1
                    results.append({
                        'input_file': pdb_file,
                        'output_file': output_file,
                        'status': 'success',
                        'length': result.get('length', 0)
                    })
                else:
                    failed += 1
                    results.append({
                        'input_file': pdb_file,
                        'status': 'failed',
                        'error': result['error']
                    })
                    
            except Exception as e:
                failed += 1
                results.append({
                    'input_file': pdb_file,
                    'status': 'failed',
                    'error': str(e)
                })
        
        batch_result = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'total_files': len(pdb_files),
            'successful': successful,
            'failed': failed,
            'results': results
        }
        
        # Save batch report.
        report_file = os.path.join(output_dir, 'batch_report.json')
        with open(report_file, 'w') as f:
            json.dump(batch_result, f, ensure_ascii=False, indent=2)
        
        batch_result['report_file'] = report_file
        return json.dumps(batch_result, ensure_ascii=False)
        
    except Exception as e:
        error_result = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'success': False,
            'error': str(e),
            'total_files': 0,
            'successful': 0,
            'failed': 0
        }
        return json.dumps(error_result, ensure_ascii=False)


def structure_to_dotbracket(
    input_path: str,
    output_path: Optional[str] = None,
    chain_filter: Optional[str] = None
) -> str:
    """Convert a PDB/CIF RNA structure to dot-bracket notation."""
    try:
        ext = os.path.splitext(input_path)[1].lower()
        tmp_pdb = os.path.splitext(input_path)[0] + "_rna_only.tmp.pdb"

        # Generate an RNA-only PDB.
        if ext in ('.cif', '.mmcif'):
            _bio_to_rna_only_pdb(input_path, tmp_pdb, is_cif=True, chain_filter=chain_filter)
        else:
            _pdb_to_rna_only_pdb(input_path, tmp_pdb, chain_filter=chain_filter)

        pdb_for_annotation = tmp_pdb

        # Prefer the RNApolis annotator CLI for command-line-consistent pairs/DB.
        sequence = ""
        db = ""
        annot_tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
                annot_tmp = tf.name
            subprocess.run([
                'annotator', '-j', annot_tmp, pdb_for_annotation
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(annot_tmp, 'r') as rf:
                data = json.load(rf)
            # Prefer bpseq.dot_bracket, matching CLI output.
            db = (
                data.get('bpseq', {})
                    .get('dot_bracket', {})
                    .get('structure', "")
            ) or data.get('dot_bracket', "") or ""
            sequence = (
                data.get('bpseq', {})
                    .get('dot_bracket', {})
                    .get('sequence', "")
            ) or ""
        except Exception:
            # Fall back to find_pairs for canonical pairs only.
            with open(pdb_for_annotation, 'r') as f:
                structure = read_3d_structure(f)
            canon, noncanon, tertiary = find_pairs(structure)
            db = _generate_dotbracket_from_canon_pairs(structure, canon)
            seq_map = { 'A':'A', 'U':'U', 'G':'G', 'C':'C' }
            sequence = ''.join(seq_map.get(res.name, 'N') for res in structure.residues)
        finally:
            if annot_tmp:
                try:
                    os.remove(annot_tmp)
                except Exception:
                    pass

        result = {
            'input_file': input_path,
            'success': True,
            'sequence': sequence,
            'dot_bracket': db,
            'length': len(sequence),
            'num_pairs': db.count('('),
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            result['output_file'] = output_path

        # Remove temporary files.
        try:
            if os.path.exists(tmp_pdb):
                os.remove(tmp_pdb)
        except Exception:
            pass

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        error_result = {
            'input_file': input_path,
            'success': False,
            'error': str(e)
        }
        return json.dumps(error_result, ensure_ascii=False)


def analyze_rna_structure(
    pdb_path: str,
    detailed_analysis: bool = True
) -> str:
    """Analyze an RNA structure in detail."""
    try:
        # Read and annotate the structure.
        # Note: annotated_structure may require a version-specific RNApolis import;
        # keep this call as-is for compatibility.
        from rnapolis.annotator import annotate_3d_structure
        structure = read_3d_structure(pdb_path)
        annotated_structure = annotate_3d_structure(structure)
        
        analysis_result = {
            'input_file': pdb_path,
            'success': True,
            'structure_info': {
                'num_chains': len(annotated_structure.chains),
                'total_residues': sum(len(chain.residues) for chain in annotated_structure.chains),
                'chains': []
            }
        }
        
        for chain in annotated_structure.chains:
            chain_info = {
                'chain_id': chain.chain_id,
                'sequence': chain.sequence,
                'sequence_length': len(chain.sequence) if chain.sequence else 0,
                'num_residues': len(chain.residues),
                'residue_types': {}
            }
            
            # Count residue types.
            for residue in chain.residues:
                res_type = residue.name
                chain_info['residue_types'][res_type] = chain_info['residue_types'].get(res_type, 0) + 1
            
            # Include base-pair information when available.
            if hasattr(chain, 'base_pairs') and chain.base_pairs:
                chain_info['num_base_pairs'] = len(chain.base_pairs)
                chain_info['base_pair_types'] = {}
                
                for bp in chain.base_pairs:
                    bp_type = f"{bp.residue1.name}-{bp.residue2.name}"
                    chain_info['base_pair_types'][bp_type] = chain_info['base_pair_types'].get(bp_type, 0) + 1
            
            analysis_result['structure_info']['chains'].append(chain_info)
        
        return json.dumps(analysis_result, ensure_ascii=False)
        
    except Exception as e:
        error_result = {
            'input_file': pdb_path,
            'success': False,
            'error': str(e)
        }
        return json.dumps(error_result, ensure_ascii=False)

