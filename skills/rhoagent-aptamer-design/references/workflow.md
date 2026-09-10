# Aptamer workflows (Tasks 4–5)

## Select inputs

- **Task 4:** single-motif Broccoli or Pepper scaffolding.
- **Task 5:** dual-motif Broccoli–Pepper scaffolding on one RNA chain for subsequent FRET testing.

Collect the user's motif/reference PDB, chain/residue selections, contig template, fixed-base mapping, desired length range, sample budget and work_dir. Collect ligand/ion context for prediction where needed. Task 5 requires both motif definitions and inter-domain constraints in a combined reference structure. Do not substitute unrelated example structures.

Example contig syntax (illustrative, only valid with matching reference residues):
- Single motif: 8+/A3-16/15+/A29-47/8+
- Dual motif: 10+/A3-16/15+/A29-47/10+/E3-22/15+/E35-47/10+

Check every referenced chain/residue exists and map fixed residues into each generated sequence. Insertion lengths change output indices; reference residue numbers must not be reused blindly.

## Default stages

| Stage / registered tool | Arguments and artifacts |
|---|---|
| 1. rdiff_scaffolding_design_tool | target_pdb_path, contigs, output_dir; explicit inference_name and selected length/sample settings. Returns merged PDB/FASTA outputs under output_dir/inference_name. |
| 2. backbone_filter | input_dir of staged scaffold PDBs, dot_bracket and optional clash_th=3.0. See the interface discrepancy below; preserve PDB/FASTA pairs and original files. |
| 3. rdiff_seq_design_tool | ss_db_constrain, input_structure_dir, output_dir, fixed_seq; num_samples explicitly chosen, optional target_seq for refinement and checkpoint overrides. |
| 4. seq_filter | input_fasta, explicit output_fasta, optional metrics_csv; GC [0.35, 0.70], homopolymer_run_len=6, remove_duplicates=true. No reference-based recovery filtering by default. |
| 5. af3_prediction_pipeline | one-record filtered FASTAs in input_dir, json_dir, output_dir; rna_id='A', selected model_seeds, ligands and gpu_id. |
| 6. scaffolding_structure_eval | pred_dir, backbone_dir, output_dir, motif_specs; enable_motif_eval='auto', optional usalign_bin. Returns results.csv and wt_motif_gate.json. |
| 7. first_round_filtering | input_csv, output_csv, filter_mode='scaffolding'; length-aware selection. |
| 8. Deliver ranked candidates | Report actual motif gates, self-consistency, compactness, contacts, scaffold lengths and candidate artifacts. |

Use critic checkpoints after major stages as described in execution.md unless the user changes them. Generic ribozyme evaluation and LLM ranking are not substitutes for the primary scaffold evaluation. A user may insert another ranking stage after validating its required columns.

## Scaffolding defaults and sample budget

The tool defaults are inference_name='scaffolding_iter', length_min=120, length_max=120, length_step=25, max_contigs_per_length=200, num_samples=5, gpu_id=0, seed=42. Collect the user's intended lengths rather than assuming 120 is biologically appropriate.

num_samples is the number per concrete contig, split across checkpoints—not the total for the entire length sweep. Explain the resulting budget before execution. Optional arguments include target_length, diffusion_num_t, checkpoint_dir/checkpoint_paths, weights_path, python_bin, rdiff_str_dir and extra_overrides. Use actual exposed arguments, not dotted conceptual names from the prompt.

## Translate scaffold outputs into sequence constraints

The default prompt says to use fixed sequence information accompanying scaffold PDBs. The sequence wrapper, however, requires explicit ss_db_constrain and fixed_seq and does not automatically select the correct sidecar for each PDB.

Read and validate each scaffold's sidecar, residue mapping and structure. Derive/obtain a compatible dot-bracket constraint and fixed mask, then invoke sequence design separately for each backbone or group with identical constraints. If the sidecar does not encode enough information, require a verified motif mapping; never guess locked positions or reuse one mask across different scaffold layouts. The default workflow-level sequence sample count is 2; the wrapper default is 10, so pass the selected value explicitly.

Sequence filtering is per FASTA. Preserve unique IDs and one-record FASTAs for AF3; aggregate/split only if adding global deduplication, and record that adaptation.

## WT motif gates and evaluation

For each motif, motif_specs can contain name, motif_ref_pdb, fixed_motif or motif_indices, wt_pred_path, and wt_fixed_motif or wt_motif_indices. Inspect the implementation's index conventions and map against actual structure residues before constructing these arrays.

Provide a matching WT prediction from the user or add an explicit WT prediction preparation stage with the same relevant ligand/ion context. Do not pretend the evaluation tool generates WT predictions itself.

Default WT gate thresholds are RMSD <= 4.0 and TM >= 0.45 when available. Keep enable_motif_eval='auto': failed/unavailable gates mean motif geometry is not validated and motif RMSD/TM should not be used as reliable selection evidence. Report each motif's gate separately. Task 5 must include both Broccoli and Pepper.

Other metrics include backbone_af3_tm_score, backbone_af3_rmsd, backbone_af3_lddt (C1'), rg_norm, end_to_end_dist_norm, motif_scaffold_contact_count and isolated_scaffold_fraction. Preserve the evaluator's actual missing values.

Scaffolding filtering defaults: top_n=100, scaffold_top_n_per_length=10, scaffold_min_backbone_tm=0.0, scaffold_min_backbone_lddt=0.0, scaffold_max_isolated_fraction=1.0; motif-scaffold contact cap is optional. These are tool defaults, not experimentally validated acceptance thresholds. Describe actual ordering and retained length diversity from the implementation/output.

## Interface discrepancy: backbone filtering

The prompt describes optional dot-bracket checks for scaffolding, but the existing backbone_filter requires dot_bracket and tests pair geometry. It is not a geometry-only filter. It also has the syntax blocker documented in setup.md.

Obtain a justified per-scaffold structure constraint and use matching groups, or offer a user-approved replacement/removal of this stage. An all-unpaired dummy string is not a faithful implementation of the original workflow. Report the discrepancy; do not silently drop the stage or claim unsupported filtering was performed.

## Optional refinement and FRET interpretation

Task 4 supports Round 2 with measured fluorescence/binding/activity feedback. Map observations to IDs, select seeds and record selection criteria, then use target_seq partial sampling on matched backbones with the motif locked. Repeat filtering, prediction and scaffold evaluation in separate Round 2 directories. There is no registered wet_lab_analysis_tool; use an available explicit analysis method.

For Task 5, report both motif gates and inter-domain organization as computational evidence. The original workflow proposes downstream F620/F507 measurements under 472 nm excitation (507 nm donor and 620 nm FRET channels). Do not infer a measured FRET ratio, fluorescence or binding activity from structural scores. Users may supply their own assay definitions and add refinement steps.

## Checkout sources

Consult rhoagent/agents/rhoagent_pipeline.py for workflow intent and rhoagent/tools/design_tools/, filtering_tools/, struc_tools/, eval_tools/ for current schemas, mappings and ranking behavior.

