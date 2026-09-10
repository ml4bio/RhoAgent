# RhoAgent design skills

Install these skills in a file-reading, command-capable agent to use RhoAgent for your own RNA designs. Start with the original workflow or ask the agent to add, remove, replace or reorder stages. The skills supply instructions; scientific software, checkpoints and compute resources must be installed separately.

| Skill folder | Design workflows |
|---|---|
| [rhoagent-ribozyme-design](rhoagent-ribozyme-design/SKILL.md) | Motif-constrained ribozymes, subsequence-constrained ribozymes, and limited-prior iterative optimization (Tasks 1–3). |
| [rhoagent-aptamer-design](rhoagent-aptamer-design/SKILL.md) | Single-motif aptamer scaffolding and dual-motif Broccoli–Pepper FRET designs (Tasks 4–5). |

## Installation

Clone https://github.com/ml4bio/RhoAgent.git or use an existing checkout. Copy the **whole selected skill folder**, including its references subdirectory, into your agent's configured skill directory. Each skill is self-contained and can be installed alone. Retain the separate RhoAgent checkout for execution.

For Codex's default personal skill location, run from the repository root:

```bash
mkdir -p ~/.codex/skills
cp -R skills/rhoagent-ribozyme-design ~/.codex/skills/
cp -R skills/rhoagent-aptamer-design ~/.codex/skills/
```

If you use a custom CODEX_HOME, use its skills directory instead. Inspect any existing destination before copying; do not overwrite a customized installation without reviewing it. For another agent, use that agent's supported skill installation directory/mechanism. Agents without automatic skill discovery can read the chosen SKILL.md explicitly. Follow your client's reload procedure if newly installed skills are not visible.

An installation request you can give another agent:

> Install the rhoagent-aptamer-design skill from the skills folder of ml4bio/RhoAgent, preserving its bundled instructions. Then help me check prerequisites for my local setup before running a design.

## Prerequisites

Install RhoAgent's Python dependencies, the RDiff components and checkpoints required for generation, and a compatible GPU environment. Prediction needs AlphaFold 3 with its model parameters, databases and executables. Structure analysis needs the relevant RNApolis/Biopython dependencies and a working USalign binary. MXfold2 is needed only if selected. Critic/LLM stages need a configured model endpoint.

The skill checks dependencies against the selected stages. Supplying predictions can remove the need to install AF3. Current eager imports can still require unrelated Python dependencies.

After installation, the agent verifies local paths and can adapt the local AF3 script template and alignment binary paths before execution. The checkout currently contains a backbone-filter indentation error that blocks tool imports; the skills identify this and explain how to proceed with a corrected checkout or an authorized repair. Installing a skill alone does not resolve runtime defects.

## Example requests

**Original ribozyme workflow**

> Use rhoagent-ribozyme-design with my sequence, secondary structure and fixed-motif mask. Follow Task 1 with the motif locked, 20 backbones and 2 sequence samples. My RhoAgent checkout is /path/to/RhoAgent and my output directory is /path/to/run.

**Subsequence-constrained design**

> Use Task 2 to retain my supplied target subsequence. Check its position and compatibility with the fixed motif before starting.

**Limited-prior optimization**

> Start Task 3 from my reference RNA sequence and functional-site positions. Infer the structural constraints, then run the original Round 1 workflow.

**Single-motif scaffolding**

> Use rhoagent-aptamer-design with my Pepper motif PDB and contig specification. Help me select a scaffold length range and run Task 4.

**Dual-motif design**

> Use Task 5 with my combined Broccoli–Pepper reference structure, both motif definitions and inter-domain constraints. Evaluate both WT motif gates.

**Customize a workflow**

> Use my existing predictions instead of running AF3, insert my candidate-filter script before evaluation, and omit final LLM ranking. Show the revised workflow and tell me which artifacts the remaining stages need.

The agent checks dependencies between stages, records modifications, and never silently restores a removed step. If removing a stage leaves required artifacts unavailable, it asks for equivalent inputs or a revised plan.

## Outputs

Expect candidate FASTAs and structures for completed generation stages, available metric/ranking tables, and a run record of settings, checkpoints, seeds, local adaptations and stage status. Computational scores do not establish experimental activity, fluorescence or FRET. No paper-specific dataset is required.

