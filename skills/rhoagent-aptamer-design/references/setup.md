# Local setup and execution

## Locate the runtime

The installed skill contains instructions, not the scientific software. Ask for or locate a RhoAgent checkout; validate it contains `rhoagent/agents/rhoagent_pipeline.py`, `rhoagent/tools/`, and `setup.py`. Use its absolute path as the working directory. Do not assume the skill's parent directory is the repository. Record the checkout revision and local modifications.

If needed, clone https://github.com/ml4bio/RhoAgent.git into a user-chosen location. Create/activate a Conda environment named `RhoAgent`, then install the checkout with `python -m pip install -e .`. The README specifies Python 3.9+, but dependency/API compatibility must be checked for the chosen interpreter rather than assumed.

## Install only what the selected stages require

| Stage | Software/resources to install and verify |
|---|---|
| RhoAgent tool access | Package dependencies from setup.py; scientific imports include NumPy, pandas, Biopython and RNApolis. Check imports after installation. |
| Backbone/scaffold generation | RDiff from https://github.com/ml4bio/RDiff; locate its RDiff_str component, compatible environment and appropriate backbone/scaffolding checkpoints. Follow its installation instructions. |
| Sequence generation | RDiff_seq component, compatible environment and sequence checkpoints. Verify main.py and configs/ribodiffusion_af3_ss.py exist. |
| AF3 prediction | AlphaFold 3 from https://github.com/google-deepmind/alphafold3, compatible GPU/CUDA environment, model parameters obtained under upstream access terms, databases and HMMER executables. Follow upstream installation guidance; a skill does not supply these assets. |
| Secondary-structure extraction/evaluation | RNApolis with the parser/annotator APIs used by struc_tools/utils/rnapolis_utils.py; Biopython and NumPy. |
| Structural alignment | USalign from https://github.com/pylelab/USalign compiled/downloaded for the local OS and architecture; verify it executes. |
| Critic / LLM ranking / built-in agent | User-selected compatible model endpoint and credentials. |
| Optional MXfold2 stage | MXfold2 executable and its runtime only if this stage is selected; inspect mxfold2_predict parameters for executable configuration. |

A supplied prediction can eliminate the AF3 runtime requirement. Removing LLM ranking does not remove the credential requirement if critic calls remain enabled. The current tools package eagerly imports other tool modules, so even a narrow workflow may require additional import-time dependencies; report these separately from stage requirements.

Do not auto-download large models/databases or start jobs merely to test installation. Check executable availability, input readability, checkpoint files, GPU compatibility and free disk space first.

## Configure local locations

Use shell exports or plain KEY=value lines in a repository-root .env:

```dotenv
RDIFF_STR_CODE_DIR=/absolute/path/to/RDiff_str
RDIFF_SEQ_CODE_DIR=/absolute/path/to/RDiff_seq
PYTHON_CONDA_RDIFF=/absolute/path/to/rdiff/environment/bin/python
RDIFF_BACKBONE_CHECKPOINT_DIR=/absolute/path/to/backbone_checkpoints
RDIFF_SCAFFOLDING_CHECKPOINT_DIR=/absolute/path/to/scaffolding_checkpoints
RDIFF_SEQUENCE_CHECKPOINT_DIR=/absolute/path/to/sequence_checkpoints
DEFAULT_MODEL=your-model
DEFAULT_API_BASE=https://your-compatible-endpoint/v1
DEFAULT_API_KEY=your-key
```

Only configure selected stages. Keep secrets out of logs and commits. The RDiff helper's own .env reader expects KEY=value, not lines beginning with export. LLM tools use process environment variables; explicitly load .env into that process where necessary. Do not assume all tools share a configuration loader.

One PYTHON_CONDA_RDIFF is used by the backbone and sequence wrappers: verify it supports both components, or explicitly adapt the local invocation/environment between stages. Checkpoint paths override checkpoint directories where supported. Record checkpoint identity and sample splits from actual returns.

## Adapt AF3 and alignment paths before executing

Inspect `rhoagent/tools/struc_tools/af3_predict.py`. Its generated shell script currently embeds server paths. After the user has installed AF3, map each to the verified local location:

- AF3 Python executable and run_alphafold.py;
- model_dir, db_dir and jax_compilation_cache_dir;
- nhmmer, hmmalign, hmmbuild, hmmsearch and jackhmmer binaries.

`script_path` / AF3_SCRIPT_PATH controls where the script is written; it does **not** configure these scientific resource paths. The tool writes and immediately executes the script, so inspecting/editing the script after calling the tool is too late. Apply targeted path changes to the local Python script template before invocation when no supported configuration exists. Preserve a diff and unrelated user edits; validate Python syntax and the corresponding shell command without running prediction. Verify shell quoting for local paths.

The bundled USalign binary may not match the user's platform. Pass `usalign_bin` to tools that expose it. The general evaluation wrapper does not expose that argument, and its DEFAULT_USALIGN_BIN is a code constant pointing at the bundled binary, not an environment setting. If needed, make a targeted local adaptation of that constant in eval_tools/utils/eval_utils.py and record it.

These are local setup adaptations, not permission to refactor the project.

## Invoke existing tools

For customized workflows, the consuming agent orchestrates existing tool classes directly. The built-in `python run_agent.py` has a fixed system prompt and may not respect skipped/reordered steps. Do not use it to silently reinstate the original workflow.

After import checks succeed, this is the existing interface (illustrative, not a new runner):

```python
import json
from rhoagent.tools import TOOL_REGISTRY

name = "seq_filter"
params = {"input_fasta": "/absolute/path/to/candidates.fasta",
          "output_fasta": "/absolute/path/to/filtered.fasta"}
tool = TOOL_REGISTRY[name]()
print(tool.parameters)  # inspect before preparing the actual call
# Execute only as an authorized stage, using the reviewed arguments:
result = tool.call(params)
payload = json.loads(result) if isinstance(result, str) else result
```

Read the installed checkout's schema and call implementation before executing: tool-name availability does not guarantee dependencies or output compatibility. Use absolute expanded paths. Inspect success/error fields, stage logs, output counts and files, not only process return codes.

## Known implementation discrepancies to check

- The supplied backbone_filter_tool.py has an indentation error in its call method. Because tools/__init__.py imports it eagerly, it can block all tool imports. Check syntax before importing. Report the exact blocker; use a corrected checkout or obtain authorization for a narrowly scoped repair. Do not claim installation succeeded.
- backbone_filter requires dot_bracket and deletes rejected PDBs from input_dir. Operate on a staged copy and keep original backbone/FASTA pairs.
- AF3 processes only the first record per FASTA file. Split retained sequences into one-record files with unique, safe IDs and preserve ID-to-backbone mappings.
- AF3's top-level success can be true when script_success is false; its shell pipeline through tee can also hide a failed predictor. Verify logs, completion and one usable prediction per expected ID.
- The prompt mentions wet_lab_analysis_tool, which is not registered here. Use an explicitly available user tool or transparent analysis of supplied measurements; do not fabricate a call.
- Broad prompt descriptions do not add missing wrapper parameters. Report unsupported options and local adaptation needs before execution.

