# Integration notes

1. Back up the repository and copy the documentation, `.gitignore`, provenance template, and privacy helper into the project root/folders.
2. Keep Study B out of the repository claims; this patch intentionally covers only the reviewed Study A code.
3. Complete `DATASET_PROVENANCE_TEMPLATE.csv` with exact Hugging Face dataset IDs, configurations, splits, immutable revisions, citations, licenses/terms, and counts. Do not invent missing identifiers.
4. Run `scripts/prepare_public_dataset.py` on a copy of the real `combined_tweets.csv`; inspect the output manually and test downstream code. The helper drops common identifiers and raw-text columns by default. It is not a guarantee of anonymization.
5. Do not upload `combined_tweets.csv`, raw tweets, model artifacts, or notebooks until a privacy/ethics review confirms they are safe and permitted.
6. Replace the placeholders in `LICENSE` only if the project owners have authority to license all included code under MIT. Dataset and pretrained-model licenses remain separate.
7. Generate a pinned dependency file from the actual successful experimental environment. No versions are guessed in this patch.
8. Correct the episode CSV logging in the actual outer training-loop file using `AUDIT_NOTES.md`; do not apply a blind edit to `train_dqn.py` if the loop is elsewhere.
9. Run every documented command from a clean clone and record outputs before claiming full reproducibility.
10. This patch has not been pushed to GitHub; apply it locally or provide the repository URL and authorized access for a direct update.
