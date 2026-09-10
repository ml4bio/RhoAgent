# 🧬 RhoAgent: Functional RNA Design Agent

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache&logoColor=white)](LICENSE)
[![RDiff](https://img.shields.io/badge/Powered%20by-RDiff-6C5CE7)](https://github.com/ml4bio/RDiff)

An agentic framework for functional RNA design, combining diffusion-based generation,
structure prediction, evaluation, and candidate filtering in a task-oriented workflow.

## 📑 Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Agent Skills](#agent-skills)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Supported Task Types](#supported-task-types)
- [Core Components](#core-components)
- [Evaluation Metrics](#evaluation-metrics)
- [Checkpoint Ensemble Behavior](#checkpoint-ensemble-behavior)
- [Tool-Call Confirmation Policy](#tool-call-confirmation-policy)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [License](#license)

<a id="key-features"></a>

## ✨ Key Features

- **Task-oriented RNA design**: the agent first asks the user to choose a task type, then loads the matching workflow.
- **Motif sampling and constrained sequence design**: ribozyme tasks support Mode 1 (fully locked fixed motif) and Mode 2 (motif sampling / unlock) via RDiff_seq partial sampling; iterative workflows can also run seed-based partial diffusion for local resampling while keeping key motifs fixed.
- **Checkpoint-ensemble design tools**: backbone, scaffolding, and sequence design tools can split a requested sample count across all checkpoints in configured checkpoint directories.
- **Ribozyme and aptamer scaffolding workflows**:
  - secondary-structure-conditioned ribozyme design
  - subsequence-conditioned ribozyme design
  - limited-prior Hovlinc optimization
  - Broccoli/Pepper single-motif scaffolding
  - Broccoli-Pepper FRET dual-motif scaffolding
- **Scaffolding-specific evaluation**: WT motif gate, backbone-AF3 self-consistency, C1'-lDDT, compactness, motif-scaffold contacts, and length-aware candidate filtering.
- **Two-mode first-round filtering**:
  - `standard`: sequence recovery, GC, sequence similarity, `ss_f1score >= 0.9` when available, TM-score ranking
  - `scaffolding`: backbone-AF3 consistency, compactness, and length-aware selection

<a id="installation"></a>

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- Conda (recommended for environment management)

### Setup

0. RDiff repository:

Please refer to the [RDiff repository](https://github.com/ml4bio/RDiff).

1. Clone the repository:
```bash
git clone <repository-url>
cd RhoAgent
```

2. Create and activate conda environment:
```bash
conda create -n RhoAgent python=3.9
conda activate RhoAgent
```

3. Install dependencies:
```bash
pip install -e .
```

4. Configure environment variables:
Create a `.env` file in the project root (or set shell exports) before running the agent.
Do not commit real API keys.

RDiff design tools read paths from `.env`. You must configure the **RDiff_str** and **RDiff_seq**
repository directories, the Python interpreter for the RDiff conda environment, and the checkpoint
directories used by each design tool.

```bash
export DEFAULT_MODEL="your-model-name"
export DEFAULT_API_BASE="https://your-openai-compatible-endpoint/v1"
export DEFAULT_API_KEY="your-api-key"

# Optional but recommended for evaluation
export DEFAULT_USALIGN_BIN="/path/to/USalign"

# Required: RDiff code repositories
# RDiff_str: structure diffusion (backbone design + motif scaffolding)
export RDIFF_STR_CODE_DIR="/path/to/RDiff_str"
# RDiff_seq: sequence design on generated backbones
export RDIFF_SEQ_CODE_DIR="/path/to/RDiff_seq"
# Python executable from the RDiff conda environment
export PYTHON_CONDA_RDIFF="/path/to/ribodiffusion/bin/python"

# Required: checkpoint directories for ensemble design
# Each directory may contain one or more .pth checkpoints.
# The requested num_samples is split across all checkpoints in the directory.
export RDIFF_BACKBONE_CHECKPOINT_DIR="/path/to/RDiff_str/backbone_checkpoints"
export RDIFF_SCAFFOLDING_CHECKPOINT_DIR="/path/to/RDiff_str/scaffolding_checkpoints"
export RDIFF_SEQUENCE_CHECKPOINT_DIR="/path/to/RDiff_seq/checkpoints"
```

Optional: set `RHOAGENT_ROOT` or `WORKSPACE_ROOT` if your `.env` file lives outside the repo root.
The agent also searches `./.env` and `<repo-root>/.env` automatically.

<a id="agent-skills"></a>

## 🧩 Agent Skills

The skills in [`skills/`](skills/README.md) provide an alternative way to use this
repository through another agent, alongside the built-in RhoAgent interface.
They guide the agent through the same design workflows, including local setup,
tool execution, evaluation, and output tracking.

| Skill | Supported workflows |
|---|---|
| [Ribozyme design](skills/rhoagent-ribozyme-design/SKILL.md) | Motif-constrained design, subsequence-constrained design, and limited-prior iterative optimization. |
| [Aptamer design](skills/rhoagent-aptamer-design/SKILL.md) | Single-motif Broccoli/Pepper scaffolding and dual-motif Broccoli–Pepper FRET scaffold design. |

Follow the original workflows or ask your agent to add, remove, replace, or reorder
steps. The skills check downstream input requirements and record workflow changes.

To get started, ask your agent:

> Install the rhoagent-ribozyme-design skill from the skills folder of
> https://github.com/ml4bio/RhoAgent. Help me install the required tools and
> configure local paths, then use the original workflow with my RNA sequence,
> secondary structure, and fixed-motif constraints.

You can also customize a run:

> Use my existing structure predictions instead of running AF3, insert my
> candidate-filter script before evaluation, and omit final LLM ranking.

Install each complete skill folder, including its supporting files. Scientific
software and model checkpoints are installed separately; the skills guide
prerequisite checks and local path adaptation. See the
[skills installation and usage guide](skills/README.md) for details.

<a id="quick-start"></a>

## 🚀 Quick Start

### Running the Agent

The simplest way to start using RhoAgent is through the interactive entry point:

```bash
conda activate RhoAgent
python run_agent.py
```

The agent will first ask you to choose one task:

1. Ribozyme Design
2. Subsequence-Conditioned Ribozyme Design
3. Hovlinc Limited-Prior Iterative Optimization
4. Single-Motif Fluorogenic Aptamer Scaffolding
5. Dual-Motif Broccoli-Pepper FRET Scaffold Design

After task selection, the agent collects only the required inputs for that task and asks for parameter confirmation.
It must also ask for confirmation before every individual tool call.

### Example Usage

```python
from rhoagent.agents.rhoagent_pipeline import build_rna_design_pipeline_agent

# Configure LLM
llm_cfg = {
    'model': 'Qwen3-235B-A22B',
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': 'your-api-key'
}

# Create agent
agent = build_rna_design_pipeline_agent(llm=llm_cfg)

# Run interactive session
messages = [{
    'role': 'user',
    'content': '''
    I want to design a functional RNA structure. Task 1.
    - wildtype_seq: ACACCCACCUGUUACAAGUCAGGACAGAAGCAGAGUAACGGUUGCUUACGCAACCGGUAAUGCUACUGGGUGU
    - target_ss: (((((((((((.[[[..{{)))).....((((..]]].(((((((....)))))))....))))}})))))))
    - fixed_motif: -----------UUACAA---------------G-GUA--------------------UAA-------------
    - work_dir: /path/to/output
    '''
}]

for response in agent.run(messages=messages):
    print(response)
```

<a id="project-structure"></a>

## 🗂️ Project Structure

```
RhoAgent/
├── rhoagent/                    # Main package
│   ├── agents/                  # Agent implementations
│   │   ├── rhoagent_pipeline.py    # RNA Design Pipeline Agent (main)
│   │   ├── assistant.py        # Base assistant agent
│   │   ├── fncall_agent.py      # Function calling agent
│   │   ├── user_agent.py        # User interaction agent
│   │   ├── doc_qa/              # Document Q&A agents
│   │   │   ├── basic_doc_qa.py
│   │   │   ├── parallel_doc_qa.py
│   │   │   └── ...
│   │   └── keygen_strategies/   # Keyword generation strategies
│   ├── tools/                   # Tool implementations
│   ├── llm/                     # LLM integration
│   ├── memory/                  # Memory management
│   │   └── memory.py
│   ├── utils/                   # Utility functions
│   │   ├── utils.py
│   │   ├── output_beautify.py
│   │   └── parallel_executor.py
│   ├── agent.py                 # Base agent class
│   ├── settings.py              # Configuration settings
│   └── multi_agent_hub.py      # Multi-agent coordination
├── examples/                    # Example scripts
│   ├── rhoagent_example.py     # Example usage
│   └── resource/                # Example resources
├── tests/                       # Test files
│   ├── agents/                  # Agent tests
│   ├── llm/                     # LLM tests
│   ├── tools/                   # Tool tests
│   └── pipeline_test_output*/   # Test output directories
├── run_agent.py                 # Main entry point
├── setup.py                     # Package configuration
├── MANIFEST.in                  # Package manifest
├── LICENSE                      # Apache 2.0 License
└── README.md                    # This file
```

<a id="supported-task-types"></a>

## 🧭 Supported Task Types

### Task 1: Ribozyme Design

For Twister-P3, CPEB3, Hammerhead, and related ribozymes.

Typical flow:

1. Confirm `wildtype_seq`, `target_ss`, `fixed_motif`, and work directory.
2. Run `rdiff_3d_backbone_design_tool`.
3. Run `backbone_filter`.
4. Run `rdiff_seq_design_tool`.
5. Run `seq_filter`.
6. Run `af3_prediction_pipeline`.
7. Run `eval_rna_struct_batch`.
8. Run `first_round_filtering(filter_mode="standard")`.
9. Run `llm_weighted_filter`.

### Task 2: Subsequence-Conditioned Ribozyme Design

Same core flow as Task 1, but the user also provides a subsequence that must be preserved.
The subsequence is merged into the fixed motif constraints before sequence design.

### Task 3: Hovlinc Limited-Prior Iterative Optimization

For Hovlinc-like cases with limited structural priors.
The agent can infer secondary structure and pocket constraints using AF3, RNApolis, and `find_pocket`.
Round 2 refinement is possible when wet-lab feedback is available.

### Task 4: Single-Motif Fluorogenic Aptamer Scaffolding

For Broccoli or Pepper motif scaffolding.
Uses `rdiff_scaffolding_design_tool`, not the generic secondary-structure backbone design tool.

Typical inputs:

- motif/reference PDB
- contig template
- scaffold length range
- ligand/ion context for AF3 when needed

Primary evaluation uses `scaffolding_structure_eval`.

### Task 5: Dual-Motif Broccoli-Pepper FRET Scaffold Design

For one-chain Broccoli-Pepper FRET constructs.
Uses a dual-motif PDB and a dual-motif contig such as:

```text
10+/A3-16/15+/A29-47/10+/E3-22/15+/E35-47/10+
```

Primary evaluation uses `scaffolding_structure_eval` with both Broccoli and Pepper motif specs.

<a id="core-components"></a>

## 🧰 Core Components

### Agents

- **RNA Design Pipeline Agent** (`rhoagent/agents/rhoagent_pipeline.py`): Main agent that orchestrates the complete design pipeline
- **Assistant** (`rhoagent/agents/assistant.py`): Base assistant with tool usage and file reading capabilities
- **FnCallAgent** (`rhoagent/agents/fncall_agent.py`): Function calling agent
- **UserAgent** (`rhoagent/agents/user_agent.py`): User interaction agent
- **Document Q&A Agents** (`rhoagent/agents/doc_qa/`): Document question-answering agents (basic and parallel)

### Tools

#### Design Tools

- **RDiff3DBackboneDesignTool** (`rdiff_3d_backbone_design_tool`): Secondary-structure-conditioned backbone generation with **RDiff_str**.
  - Requires `RDIFF_STR_CODE_DIR` and `PYTHON_CONDA_RDIFF` in `.env`.
  - Reads checkpoints from `RDIFF_BACKBONE_CHECKPOINT_DIR` unless `checkpoint_dir` or `checkpoint_paths` is passed.
  - Splits total `num_samples` across all checkpoints.
- **RDiffScaffoldingDesignTool** (`rdiff_scaffolding_design_tool`): Motif scaffolding using **RDiff_str** `rna_scaffolding_training` + `scaffolding_iter`.
  - Requires `RDIFF_STR_CODE_DIR` and `PYTHON_CONDA_RDIFF` in `.env`.
  - Reads checkpoints from `RDIFF_SCAFFOLDING_CHECKPOINT_DIR`.
  - Inputs include `target_pdb_path`, `contigs`, length range, and total `num_samples`.
  - Merges shard outputs into `output_dir/inference_name`.
- **RDiffSeqDesignTool** (`rdiff_seq_design_tool`): Sequence design on filtered backbones with **RDiff_seq**.
  - Requires `RDIFF_SEQ_CODE_DIR` and `PYTHON_CONDA_RDIFF` in `.env`.
  - Reads checkpoints from `RDIFF_SEQUENCE_CHECKPOINT_DIR`.
  - Splits total `num_samples` across all sequence checkpoints.

#### Structure Tools
- **AF3PredictionPipelineTool**: Batch structure prediction using AlphaFold3
- **ExtractSSTool**: Extract secondary structure (dot-bracket) from PDB/CIF files
- **FindPocketTool**: Identify functional pocket regions from structure and seed positions
- **Mxfold2PredictTool**: Predict secondary structure from sequence
- **AnalyzeRNAStructureTool**: Analyze RNA structure and return chain, residue, and base pair statistics

#### Evaluation Tools
- **EvalRNAStructBatchTool**: Batch evaluation of predicted structures
  - TM-score and RMSD (structural similarity)
  - Sequence recovery rate and sequence similarity
  - Secondary structure F1-score (ss_f1score)
  - Functional core F1-score (functional_core_f1score)
  - Canonical pair ratios (`ss_pairs_ratio`, `designed_pseudoknot_canonical_pairs_ratio`)
- **ScaffoldingStructureEvalTool**: Scaffolding-specific AF3 evaluation
  - WT motif gate
  - motif RMSD/TM only when WT motif gate passes
  - backbone-AF3 TM/RMSD/lDDT
  - compactness and motif-scaffold contact metrics
- **FirstRoundFilterTool**: First-round filtering based on multiple criteria
  - `filter_mode="standard"` for ribozyme/general evals
  - `filter_mode="scaffolding"` for length-aware scaffolding selection
- **LLMWeightedFilterTool**: LLM-determined weighted filtering for final candidate selection
- **PipelineCriticTool**: Quality evaluation and decision-making for pipeline steps

#### Filtering Tools
- **BackboneFilterTool**: Filter backbone structures based on secondary structure matching
- **SeqFilterTool**: Filter FASTA records using GC, homopolymer, duplicate, optional repeated subsequence, and optional recovery-rate criteria

<a id="evaluation-metrics"></a>

## 📊 Evaluation Metrics

The evaluation system provides comprehensive metrics for RNA design assessment:

### Structure Metrics
- **TM-score**: Structural similarity score (0-1, higher is better)
- **RMSD**: Root mean square deviation (lower is better)

### Sequence Metrics
- **seq_rec_rate**: Sequence recovery rate (percentage of identical positions)
- **seq_similarity**: Sequence similarity using local alignment (0-1)

### Secondary Structure Metrics
- **ss_f1score**: F1-score for secondary structure prediction (0-1)
- **functional_core_f1score**: F1-score for fixed motif regions (0-1)
- **ss_pairs_ratio**: Ratio of canonical pairs (AU+GC+GU) in target structure
- **designed_pseudoknot_canonical_pairs_ratio**: Ratio of canonical AU/GC/GU pairs in designed/predicted pseudoknot regions

### Scaffolding Metrics

- **backbone_af3_tm_score / backbone_af3_rmsd**: AF3 prediction self-consistency against generated backbone
- **backbone_af3_lddt**: C1'-based lDDT after backbone alignment
- **rg_norm / end_to_end_dist_norm**: Length-normalized compactness metrics
- **non_motif_long_range_contact_density**: Contacts among non-motif scaffold residues
- **isolated_scaffold_fraction**: Fraction of weakly connected scaffold residues
- **motif_scaffold_contact_count**: Potential unintended long-range motif-scaffold contacts
- **motif_rmsd / motif_tm_score**: Used only when the WT motif gate indicates AF3 can reproduce the wild-type motif

<a id="checkpoint-ensemble-behavior"></a>

## 🎯 Checkpoint Ensemble Behavior

The design tools support running multiple checkpoints automatically. The requested `num_samples`
is treated as the total number of samples and split approximately evenly across checkpoints in the
configured directory.

Example:

```bash
export RDIFF_SEQUENCE_CHECKPOINT_DIR=/path/to/RDiff_seq/checkpoints
```

If that directory contains four `.pth` files and the tool receives `num_samples=100`,
each checkpoint is assigned about 25 samples. Exact allocation uses integer round-robin splitting.

You can override the `.env` directory per call using:

- `checkpoint_dir`
- `checkpoint_paths`

<a id="tool-call-confirmation-policy"></a>

## 🔐 Tool-Call Confirmation Policy

The pipeline agent is intentionally conservative. Before any tool call, it must show:

1. the exact registered tool name
2. the exact JSON parameters
3. a prompt asking the user to confirm or modify the call

The agent should not call the tool until the user confirms the parameters.


<a id="dependencies"></a>

## 📚 Dependencies

### Core Dependencies
- `dashscope>=1.11.0`: Alibaba Cloud DashScope API
- `pydantic>=2.3.0`: Data validation
- `openai`: OpenAI-compatible API support
- `requests`, `json5`, `jsonschema`: HTTP and JSON handling
- `Bio`: Biopython for biological data processing
- `numpy`, `pandas`: Numerical computing and data manipulation

### Project Files

- `run_agent.py`: Main entry point for running the RNA design agent interactively
- `setup.py`: Package configuration and dependencies
- `MANIFEST.in`: Files to include in package distribution
- `.gitignore`: Git ignore patterns

<a id="citation"></a>

## 📄 Citation

If you use RhoAgent in your research, please cite the associated paper:

```bibtex
@article{rhoagent,
  title   = {Programmable functional RNA design with a virtual scientist},
  author  = {Han Huang, Tianqing Zhang, Liang Hong, Ziqian Lin, Dongchen He, Jiuming Wang, Tian-Ying Wu, Huimin Li, Da Han, Le Yu, Jiyue Jiang, Yimin Fan, Yanruisheng Shao, Irwin King, Pengan Chen, Chun Kit Kwok, James J. Collins, and Yu Li},
  journal = {Nature Biotechnology},
  year    = {2026},
  doi     = {10.1038/s41587-026-03362-3}
}
```

<a id="license"></a>

## 📜 License

RhoAgent is released under the [Apache License 2.0](LICENSE).
