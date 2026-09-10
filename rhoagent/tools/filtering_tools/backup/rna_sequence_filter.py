#!/usr/bin/env python3
"""
RNA Sequence Filter

This script filters RNA sequences based on multiple criteria:
1. GC content between 0.3 and 0.75
2. No duplicate sequences
3. No more than 6 consecutive identical bases
4. No repeating subsequences longer than 6 bases
"""

import argparse
import sys
from collections import defaultdict
from typing import List, Tuple, Set
import pdb

def read_fasta(file_path: str) -> List[Tuple[str, str]]:
    """
    Read sequences from a FASTA file.
    
    Args:
        file_path: Path to the FASTA file
        
    Returns:
        List of tuples containing (name, sequence)
    """
    sequences = []
    current_name = ""
    current_sequence = ""
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_name and current_sequence:
                        sequences.append((current_name, current_sequence.upper()))
                    # Start new sequence
                    current_name = line[1:]  # Remove '>' character
                    current_sequence = ""
                else:
                    current_sequence += line
            
            # Don't forget the last sequence
            if current_name and current_sequence:
                sequences.append((current_name, current_sequence.upper()))
                
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    return sequences


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a sequence.
    
    Args:
        sequence: RNA sequence
        
    Returns:
        GC content as a fraction (0.0 to 1.0)
    """
    if not sequence:
        return 0.0
    
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def has_consecutive_repeats(sequence: str, max_consecutive: int = 6) -> bool:
    """
    Check if sequence has more than max_consecutive consecutive identical bases.
    
    Args:
        sequence: RNA sequence
        max_consecutive: Maximum allowed consecutive identical bases
        
    Returns:
        True if sequence has too many consecutive repeats
    """
    if len(sequence) <= max_consecutive:
        return False
    
    current_base = sequence[0]
    current_count = 1
    
    for base in sequence[1:]:
        if base == current_base:
            current_count += 1
            if current_count > max_consecutive:
                return True
        else:
            current_base = base
            current_count = 1
    
    return False


def has_repeating_subsequences(sequence: str, max_length: int = 6) -> bool:
    """
    Check if sequence has repeating subsequences longer than max_length.
    
    Args:
        sequence: RNA sequence
        max_length: Maximum allowed length of repeating subsequences
        
    Returns:
        True if sequence has repeating subsequences longer than max_length
    """
    if len(sequence) <= max_length * 2:
        return False
    
    # Check for repeating subsequences of length max_length + 1
    for length in range(max_length + 1, len(sequence) // 2 + 1):
        subsequences = set()
        for i in range(len(sequence) - length + 1):
            subseq = sequence[i:i + length]
            if subseq in subsequences:
                return True
            subsequences.add(subseq)
    
    return False


def filter_sequences(sequences: List[Tuple[str, str]], 
                    min_gc: float = 0.3, 
                    max_gc: float = 0.75,
                    max_consecutive: int = 6,
                    max_subseq_length: int = 6) -> List[Tuple[str, str]]:
    """
    Filter sequences based on all criteria.
    
    Args:
        sequences: List of (name, sequence) tuples
        min_gc: Minimum allowed GC content
        max_gc: Maximum allowed GC content
        max_consecutive: Maximum allowed consecutive identical bases
        max_subseq_length: Maximum allowed length of repeating subsequences
        
    Returns:
        Filtered list of sequences
    """
    filtered_sequences = []
    seen_sequences = set()
    
    for name, sequence in sequences:
        # Check for duplicate sequences
        if sequence in seen_sequences:
            print(f"Skipping {name}: Duplicate sequence")
            continue
        
        # Check GC content
        gc_content = calculate_gc_content(sequence)
        if gc_content < min_gc or gc_content > max_gc:
            print(f"Skipping {name}: GC content {gc_content:.3f} outside range [{min_gc}, {max_gc}]")
            continue
        
        # Check for consecutive repeats
        if has_consecutive_repeats(sequence, max_consecutive):
            print(f"Skipping {name}: Has more than {max_consecutive} consecutive identical bases")
            continue
        
        # Check for repeating subsequences
        if has_repeating_subsequences(sequence, max_subseq_length):
            print(f"Skipping {name}: Has repeating subsequences longer than {max_subseq_length}")
            continue
        
        # All checks passed
        filtered_sequences.append((name, sequence))
        seen_sequences.add(sequence)
        print(f"Keeping {name}: GC={gc_content:.3f}")
    
    return filtered_sequences


def write_fasta(sequences: List[Tuple[str, str]], output_file: str):
    """
    Write sequences to a FASTA file.
    
    Args:
        sequences: List of (name, sequence) tuples
        output_file: Output file path
    """
    try:
        with open(output_file, 'w') as f:
            for name, sequence in sequences:
                f.write(f">{name}\n")
                f.write(f"{sequence}\n")
        print(f"Successfully wrote {len(sequences)} sequences to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Filter RNA sequences based on multiple criteria")
    parser.add_argument("input_file", help="Input FASTA file")
    parser.add_argument("output_file", help="Output FASTA file")
    parser.add_argument("--min-gc", type=float, default=0.3, 
                       help="Minimum GC content (default: 0.3)")
    parser.add_argument("--max-gc", type=float, default=0.75, 
                       help="Maximum GC content (default: 0.75)")
    parser.add_argument("--max-consecutive", type=int, default=6,
                       help="Maximum consecutive identical bases (default: 6)")
    parser.add_argument("--max-subseq-length", type=int, default=7,
                       help="Maximum length of repeating subsequences (default: 6)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Read sequences
    print(f"Reading sequences from {args.input_file}...")
    sequences = read_fasta(args.input_file)
    print(f"Found {len(sequences)} sequences")
    
    if not sequences:
        print("No sequences found in input file.")
        sys.exit(1)
    
    # Filter sequences
    print("Filtering sequences...")
    if args.quiet:
        # Redirect print to /dev/null for quiet mode
        import os
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    filtered_sequences = filter_sequences(
        sequences, 
        min_gc=args.min_gc,
        max_gc=args.max_gc,
        max_consecutive=args.max_consecutive,
        max_subseq_length=args.max_subseq_length
    )
    
    if args.quiet:
        sys.stdout.close()
        sys.stdout = original_stdout
    
    # Write output
    print(f"Filtered {len(sequences)} sequences down to {len(filtered_sequences)} sequences")
    write_fasta(filtered_sequences, args.output_file)


if __name__ == "__main__":
    main() 