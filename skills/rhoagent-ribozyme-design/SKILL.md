---
name: rhoagent-ribozyme-design
description: "Design functional RNAs with RhoAgent's ribozyme workflows: motif-constrained, subsequence-constrained, or limited-prior iterative design. Use for local setup, execution or user-customized versions of these workflows."
---

# RhoAgent ribozyme design

Tasks 1–3: use the repository's existing ribozyme workflows as defaults, then honor user-requested stage changes.

## Procedure

1. Read [workflow instructions](references/workflow.md) to select the task, identify required inputs and present the original workflow.
2. Read [execution and customization](references/execution.md) before adding, removing, replacing or reordering stages. Carry the user's choices through execution; check artifact dependencies.
3. Read [local setup](references/setup.md) before execution or installation. Resolve the runtime checkout and install/check only selected-stage dependencies, plus unavoidable import-time dependencies. Adapt local AF3/alignment paths where needed.
4. Use the checkout's actual tool schemas and call implementations. Present concrete arguments and respect the agreed execution authorization. Check returned artifacts before advancing.
5. Deliver candidate artifacts, available evaluations and a run record with all deviations, skipped stages and remaining blockers.

The skill is independently installable; its instructions do not require sibling skills. It requires a separate RhoAgent checkout and scientific software. No paper-specific dataset is required. Do not report an omitted or failed stage as completed.

