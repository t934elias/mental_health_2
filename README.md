# Mental-health conversational RL research code

## Scope

This repository snapshot documents the **Study A / legacy DQN pipeline only**. It does not claim to contain or reproduce a second study, a latent-ground-truth-risk simulator, oracle or stepped-care baselines, bootstrap-head Q-learning, Lagrangian optimization, or robustness/deferral experiments.

The code reviewed for this documentation contains PyTorch models and training code, NumPy-based data handling, transformer-based text models, and an LLM-prompted conversational user simulator. It is therefore inaccurate to describe the reviewed implementation as wholly NumPy-based.

## Code map (provisional)

| Component | Files reviewed | Role supported by code |
|---|---|---|
| State construction | `state_builder(1).py`, `state_generation.py` | Builds model-derived state representations from text/features |
| Risk and emotion models | `risk_model.py`, `train_risk.py`, `train_emotion.py`, `train_depression.py`, `generate_labels_risk2.py` | Predictive text-model training/inference and label/data preparation |
| Policy learning | `policy_engine.py`, `policy_model.py`, `train_policy(1).py`, `train_dqn.py` | Seven-action policy/DQN-related components |
| Conversational interaction | `user_simulation.py` | LLM-prompted simulated user turns; not an explicit hidden-state risk simulator |
| Evaluation / baselines | `random_policy_baseline.py`, `ablation.py` | Random-policy comparison and policy feature ablations |
| Training/evaluation utilities | `evaluate_agent.py` if present in the full repository; training-loop script if separate | Keep with Study A only after matching imports and execution paths |

Keep the original filenames unless all imports and documentation are updated together. The list is based on the files supplied for review, not a guarantee that it exhausts the remote repository.

## Data

All datasets used in the experiments are described by the project team as publicly accessible through Hugging Face. For reproducibility, the final repository must identify each exact Hugging Face dataset repository ID, configuration, split, revision/commit, and the exact subset/preprocessing used. A generic statement that data are “on Hugging Face” is not enough to retrieve the same data later.

Do not commit copies of social-media posts or account identifiers merely because the source dataset is public. See [DATA_AND_ETHICS.md](DATA_AND_ETHICS.md). Use the data card and upstream terms for each dataset; do not assume the software license in this repository also licenses the datasets.

## Installation and execution

The reviewed files import packages including PyTorch, NumPy, pandas, scikit-learn, Transformers, Sentence Transformers, and related dependencies. Exact compatible versions were not established from the supplied files. A tested, pinned environment file must be generated from the environment used for the reported experiments; this README intentionally does not invent dependency versions.

Before describing a command as reproducible, test it from a clean clone and environment and record:

1. the command and working directory;
2. required input datasets and exact Hugging Face IDs/revisions;
3. required model checkpoints and scalers;
4. expected outputs and their schemas;
5. random seeds and hardware/software environment;
6. a successful clean-run result.

## Reproducibility statement

Do not claim that the repository reproduces every number, table, and figure until every reported result has a documented command, data source, model artifact, and clean-run verification. Distinguish code availability from data availability and from actual end-to-end reproducibility.

## Citation

Add the final paper citation and the exact repository release/commit used for the experiments before submission.
