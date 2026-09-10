"""
MXfold2 utility functions for RNA secondary-structure prediction.
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def parse_mxfold2_output(output: str) -> Dict[str, Any]:
    """
    Parse MXfold2 command output.
    
    Args:
        output: Raw stdout from the MXfold2 command.
        
    Returns:
        Parsed prediction dictionary.
    """
    lines = output.strip().split('\n')
    results = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check whether this is a sequence identifier line.
        if line.startswith('>'):
            seq_id = line[1:]  # Remove the '>' prefix.
            
            # The next line should contain the sequence.
            if i + 1 < len(lines):
                sequence = lines[i + 1].strip()
                
                # The following line should contain the structure.
                if i + 2 < len(lines):
                    structure_line = lines[i + 2].strip()
                    
                    # Structure lines usually look like: ((((....)))) (score)
                    if '(' in structure_line and ')' in structure_line:
                        # Split structure and score.
                        parts = structure_line.rsplit(' ', 1)
                        if len(parts) == 2:
                            structure = parts[0]
                            try:
                                score = float(parts[1].strip('()'))
                            except ValueError:
                                score = None
                        else:
                            structure = structure_line
                            score = None
                    else:
                        structure = structure_line
                        score = None
                    
                    results.append({
                        'sequence_id': seq_id,
                        'sequence': sequence,
                        'structure': structure,
                        'score': score,
                        'length': len(sequence)
                    })
                    
                    i += 3  # Skip lines that were already processed.
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return {
        'predictions': results,
        'total_sequences': len(results)
    }


def predict_secondary_structure(
    fasta_file: str,
    output_dir: str = None,
    mxfold2_bin: str = 'mxfold2'
) -> Dict[str, Any]:
    """
    Predict RNA secondary structure with MXfold2.
    
    Args:
        fasta_file: Input FASTA file path.
        output_dir: Optional output directory.
        mxfold2_bin: Path to the MXfold2 executable.
        
    Returns:
        Dictionary containing prediction results.
    """
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA file does not exist: {fasta_file}")
    
    # Check whether MXfold2 is available.
    try:
        subprocess.run([mxfold2_bin, '--help'], 
                      capture_output=True, 
                      check=True, 
                      timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(f"Cannot find or execute MXfold2: {mxfold2_bin}")
    
    # Read the FASTA file to validate its format.
    try:
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        if not sequences:
            raise ValueError("No sequences found in FASTA file")
    except Exception as e:
        raise ValueError(f"Failed to parse FASTA file: {e}")
    
    # Run MXfold2 prediction.
    try:
        cmd = [mxfold2_bin, 'predict', fasta_file]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse output.
        predictions = parse_mxfold2_output(result.stdout)
        
        # Add metadata.
        predictions['input_file'] = os.path.abspath(fasta_file)
        predictions['mxfold2_version'] = 'mxfold2'
        predictions['command'] = ' '.join(cmd)
        
        # Save results when an output directory is provided.
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'mxfold2_predictions.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            predictions['output_file'] = output_file
        
        return predictions
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MXfold2 execution failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("MXfold2 execution timed out")
    except Exception as e:
        raise RuntimeError(f"MXfold2 execution error: {e}")


# def validate_fasta_file(fasta_file: str) -> Dict[str, Any]:
#     """
#     Validate FASTA file format and content.
    
#     Args:
#         fasta_file: FASTA file path.
        
#     Returns:
#         Validation result dictionary.
#     """
#     try:
#         sequences = list(SeqIO.parse(fasta_file, "fasta"))
        
#         validation_result = {
#             'valid': True,
#             'total_sequences': len(sequences),
#             'sequences': []
#         }
        
#         for seq in sequences:
#             seq_info = {
#                 'id': seq.id,
#                 'description': seq.description,
#                 'length': len(seq.seq),
#                 'sequence': str(seq.seq)
#             }
            
#             # Check whether the sequence contains only RNA bases.
#             rna_bases = set('AUCG')
#             seq_bases = set(str(seq.seq).upper())
#             if not seq_bases.issubset(rna_bases):
#                 seq_info['warning'] = f"Sequence contains non-RNA bases: {seq_bases - rna_bases}"
            
#             validation_result['sequences'].append(seq_info)
        
#         return validation_result
        
#     except Exception as e:
#         return {
#             'valid': False,
#             'error': str(e),
#             'total_sequences': 0,
#             'sequences': []
#         }



