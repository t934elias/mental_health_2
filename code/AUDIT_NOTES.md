# Repository audit notes (non–Study B items)

## Findings supported by the files supplied

- The reviewed implementation uses PyTorch for neural models/training and NumPy for data/state operations. Do not call the full implementation “all NumPy.”
- `user_simulation.py` is an LLM-prompted conversational user simulation used with model-estimated states. It does not show an independent latent true-risk variable or a specified risk-transition process.
- `rl1.py` defines seven actions, reward logic, terminal conditions, and DQN configuration. Reward changes are based on model-predicted risk/state changes; they should not be described as measured clinical improvement.
- `ablation.py` evaluates policy feature ablations. Its reported results should be accompanied by a precise description of training/evaluation splits and seeds. The reviewed version appears to retrain the full policy variant rather than load a pretrained checkpoint for the claimed pretrained condition (the checkpoint-loading call is commented out).
- `state_generation.py` reads a local `combined_tweets.csv` and creates a state dataset. The actual `combined_tweets.csv` was not supplied for inspection, so its row count, columns, usernames, and privacy status remain unverified.
- The reviewed source set does not establish the complete corpus size or prove that all paper tables/figures can be regenerated.

## Training log correction to make in the actual training-loop file

The training-loop snippet previously supplied for review declared five CSV columns (`episode`, `total_reward`, `avg_loss`, `avg_risk`, `avg_uncertainty`) but wrote only three values, and passed the episode statistics dictionary as one CSV cell. The `DQNTrainer.summarize_episode` implementation in `train_dqn.py` confirms these values are available as:

```python
avg_loss = np.mean(stats["losses"]) if stats["losses"] else 0.0
avg_risk = np.mean(stats["risks"]) if stats["risks"] else 0.0
avg_uncertainty = np.mean(stats["uncertainties"]) if stats["uncertainties"] else 0.0
total_reward = stats["episode_reward"]
```

Write one scalar per header, for example:

```python
writer.writerow([
    episode + 1,
    total_reward,
    avg_loss,
    avg_risk,
    avg_uncertainty,
])
```

Also update the `file_exists` flag immediately after writing the header (or check file existence at the point of writing) so the header is not repeated on every episode. Apply this to the actual training-loop file only after confirming it is the same snippet; the supplied `train_dqn.py` is the trainer class, not necessarily that outer loop.

## Items that cannot be completed from the supplied materials

- Inspect and redact the real `combined_tweets.csv`: the file was not supplied.
- Verify the 10,159-row corpus and its exact Hugging Face repository/revision: the CSV and exact dataset ID were not supplied.
- Pin dependencies to the real experimental environment: environment lockfile/version output was not supplied.
- Verify all manuscript numbers, tables, figures, checkpoints, and execution commands: full manuscript outputs and all artifacts were not supplied.
- Push changes to the remote repository: no repository URL or authenticated repository connection was provided.

These are evidence gaps, not reasons to invent values or fabricate files.
