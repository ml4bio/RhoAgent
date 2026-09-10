# Workflow execution and customization

## Plan from the default

Read workflow.md and choose the matching task. If the user's request already identifies it, do not ask them to choose it again. Collect only missing inputs and relevant preferences. Present the original stages, then the user's edits, with each stage's inputs, outputs and dependencies.

The repository's default interaction asks for the exact tool name and JSON arguments to be confirmed before each tool call, including the critic. Describe and use that behavior for the unchanged workflow; honor explicit user changes to execution/confirmation preferences within their authorization.

Retain critic checkpoints after each major default stage. Call critic_agent_tool with step_summary containing step_name, status (success/failed/partial), key_metrics and suggestion_for_critic. ACCEPT_AND_CONTINUE proceeds; RETRY_THIS_STEP permits one explained retry; ABORT_PIPELINE stops. If the critic itself fails, report it rather than inventing a decision. If the user removes critic stages, record the change and do not call them.

## Editing the workflow

Treat the plan as an artifact dependency graph, not a mandatory checklist. Allow insert/delete/replace/reorder requests:

| Request | Required handling |
|---|---|
| Skip AF3; use existing predictions | Verify candidate IDs, sequences, chains and PDB/CIF files; map them to backbones/FASTA. Check ligand context when relevant. Do not install or invoke AF3 for this stage. |
| Remove optional final ranking | Deliver the preceding CSV/candidates with the actual ordering and explain that LLM weighting was not applied. Keep the critic only if still selected. |
| Insert a user-defined filter | Inspect its command/API, input/output contract, thresholds, candidate IDs and dependencies; place it where its required artifacts exist. Record invocation and candidate losses. |
| Remove evaluation but retain ranking | Inspect required CSV columns. Use compatible user-supplied metrics, select an agreed alternative ranking, or stop that branch with missing fields listed. Never synthesize scores. |
| Reorder dependent steps | Explain the unavailable inputs; accept the order if the user supplies equivalent artifacts, otherwise request a feasible order. |
| Stop after generation | Return generated artifacts and explicitly label prediction/evaluation as not performed. |

Never restore a deliberately omitted stage. Do not force paper datasets or sequences. Unsupported inserted stages require an available implementation or a separate agreed implementation task.

Validate RNA alphabet, sequence/structure/motif lengths, pseudoknot bracket pairing, residue/chain mapping and fixed bases before expensive calls. Conflicting constraints need resolution, not silent overwrites. Avoid losing structure/sequence provenance when renaming or staging artifacts.

## Run records and outputs

Create a new user-selected work directory; avoid overwriting prior results. Follow default stage numbering when useful: 01_backbone, 02_sequence, 02_sequence_filtered, 03_af3_json, 04_structure_pred, 05_evaluation. Use separate round2 directories.

Record the selected task and stage order; additions/skips/replacements; exact non-secret arguments; source revision and local patches; dependency versions; checkpoint hashes or paths/identity; supported seeds and unsupported seed controls; artifact mappings; stage status and counts; critic decisions and retry changes. Preserve logs and output paths. Do not promise deterministic identity across hardware/software or uncontrolled seeds.

Stop dependent stages on empty output, missing artifacts, failed tool returns or incompatible metrics. Report completed, skipped and blocked stages separately. Distinguish structural plausibility from measured catalytic activity, binding, fluorescence or FRET. Deliver candidate sequences/structures, available metric tables, selection criteria and concise limitations.

