# Ribozyme workflows (Tasks 1–3)

## Select inputs

- **Task 1: motif-constrained ribozyme design.** Typical targets include Twister-P3, CPEB3 and Hammerhead; users provide their own sequences and constraints.
- **Task 2: subsequence-constrained design.** Task 1 plus a specified subsequence that must be preserved.
- **Task 3: limited-prior optimization.** Hovlinc-style design using a reference sequence and functional-site positions, with optional experimental-feedback refinement.

Path A uses wildtype_seq, target_ss, fixed_motif and work_dir. In fixed_motif, '-' means redesignable; letters are locked bases. Validate equal lengths and motif compatibility before generation. Path B takes wildtype_seq and cut_positions, predicts a reference structure, extracts secondary structure and infers a pocket mask. Task 2 defaults to A; Task 3 defaults to B with an optional expert structure hint.

Task 1/2 Mode 1 keeps the motif locked. Mode 2 unlocks only explicitly selected motif positions; record the revised mask and preserve protected catalytic positions. The wrapper exposes fixed_seq and target_seq, not a generic motif_sampling flag.

Task 2's original prompt merges the subsequence into the motif **prefix** and checks wildtype alignment. Confirm the intended position; a non-prefix placement is a custom constraint adaptation. Reject conflicts with locked bases rather than silently overwriting them.

## Default path B preparation

1. Create one-record WT FASTA; run af3_prediction_pipeline(input_dir, json_dir, output_dir), with explicit rna_id, model_seeds and ligands when relevant.
2. Run batch_pdb_to_dotbracket(input_dir, output_dir, file_pattern). Its default is '*.pdb'; use '*.cif' for AF3 CIF inputs and stage files appropriately.
3. Run find_pocket(structure_path, seed_positions=cut_positions). Defaults are round1_threshold=14.0, round2_threshold=9.5, exclude_window=5, max_rounds=2.
4. Review extracted target_ss and the returned pocket motif with the user before generation. find_pocket uses the first chain and structure residue numbering: ensure that this is the intended RNA and map sequence positions explicitly.

## Default design stages

The workflow-level defaults are **20 backbones and 2 sequence samples**, whereas both wrappers default to 10 if omitted. Pass 20 and 2 explicitly for fidelity; users can choose other counts.

| Stage / registered tool | Arguments and artifacts |
|---|---|
| 1. rdiff_3d_backbone_design_tool | ss_db_constrain=target_ss, output_dir, num_samples=20, gpu_id=0, seed=42; optional checkpoint_dir/checkpoint_paths. Returns checkpoint shards and generated PDB paths. |
| 2. backbone_filter | input_dir containing staged PDBs, dot_bracket=target_ss, clash_th=3.0. Filters in place; preserve originals and inspect the setup blocker first. |
| 3. rdiff_seq_design_tool | ss_db_constrain=target_ss, input_structure_dir, output_dir, fixed_seq=fixed_motif, num_samples=2; optional target_seq, batch_size=10, gpu_id=0 and checkpoint overrides. Produces per-candidate FASTAs and checkpoint shard FASTAs. |
| 4. seq_filter | input_fasta and explicit output_fasta in 02_sequence_filtered; optional metrics_csv. GC [0.35, 0.70], homopolymer_run_len=6, remove_duplicates=true. Do not supply reference_sequence/reference_fasta by default. |
| 5. af3_prediction_pipeline | input_dir containing one-record filtered FASTAs, json_dir, output_dir; optional rna_id='A', model_seeds, ligands, gpu_id and script_path. Produces AF3 inputs, script/log and predictions. |
| 6. eval_rna_struct_batch | gt_struc_dir=filtered backbones, gt_seq=wildtype_seq, pred_struc_dir=predictions, pred_seq_dir=filtered FASTAs, target_ss, fixed_motif and output_dir. Produces evaluation results.csv. |
| 7. first_round_filtering | input_csv=results.csv, output_csv=results_filtered.csv, filter_mode='standard'. |
| 8. llm_weighted_filter | input_csv=results_filtered.csv, output_csv, top_k=20; optional model and explicit weight_tm_score/weight_seq_rec_rate/weight_reason. |

After each major stage use the critic behavior described in execution.md unless the user changes it.

Backbone generation distributes the total count across checkpoints and increments seeds by shard. Its output root contains shard directories; backbone_filter and sequence design need the actual PDB-containing directories. Iterate shards or stage their files with an explicit mapping, not a blind call on the root. Sequence n_samples is split across checkpoints and applied to input backbones, so it is not necessarily the final FASTA count. The sequence wrapper does not expose seed or a diffusion-depth argument; record this limitation.

Sequence filtering is per input FASTA. Deduplication within a one-record file does not deduplicate the whole run: if global deduplication is desired, aggregate records for filtering, then split retained records for AF3 and record the adaptation. Do not feed the aggregated shard FASTA alongside per-candidate FASTAs.

Default standard filtering keeps seq_rec_rate < 0.5, GC in [0.3, 0.7], seq_similarity < 0.7 when present, and ss_f1score >= 0.9 when the column exists. It sorts by descending tm_score and keeps top_n=100. These GC limits intentionally differ from the preceding sequence-filter defaults.

The prompt suggests TM/sequence weights around 0.6–0.7 / 0.3–0.4; the implementation obtains weights from the LLM unless explicitly supplied. Do not report those suggested ranges as fixed executed values. Record returned weights and rankings.

Evaluate backbone-prediction self-consistency, RMSD, sequence recovery/similarity, secondary structure and functional-core F1, and pseudoknot recall where available. Generated backbones used as gt_struc_dir are reference designs, not experimental ground truth. Sequence recovery is a fraction in these CSVs.

## Task 3 refinement

Run Path B and stages 1–8 for Round 1. If measured activity data are supplied, map measurements to exact candidate IDs, select supported high-activity seeds, and document how low-activity variants influence selection. Run sequence resampling on corresponding backbones using target_seq for partial sampling while preserving fixed_seq, then repeat sequence filtering, AF3, evaluation and ranking in separate Round 2 outputs.

Do not invent a wet_lab_analysis_tool. Derive any supplied ranking weights transparently from an available analysis method and record it. Without measurements, deliver Round 1 candidates. User-directed computational refinement without wet-lab data is allowed, but label it accordingly.

## Useful checkout entry points

Read the installed revision of agents/rhoagent_pipeline.py under rhoagent/ for default workflow intent, and tools/design_tools/, tools/filtering_tools/, tools/struc_tools/ and tools/eval_tools/ for schemas and behavior. Resolve discrepancies in favor of executable interfaces while clearly recording any deviation from the described workflow.

