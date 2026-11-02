# Data provenance, access, privacy, and ethics

## Dataset access

The project team states that the datasets used are publicly available on Hugging Face. The reviewed source files do not consistently identify the exact Hugging Face repository IDs, configurations, splits, or immutable revisions. Add one row per dataset to `DATASET_PROVENANCE_TEMPLATE.csv` and complete every field before submission.

For each dataset, record:

- exact Hugging Face dataset ID and dataset-card URL;
- configuration and split;
- immutable commit/revision (not only `main`);
- upstream citation and original source;
- dataset-card license and any separate platform terms;
- number of rows downloaded, retained, and used in each experiment;
- filtering, deduplication, preprocessing, and label-generation steps;
- retrieval command/script and expected schema.

Hugging Face dataset cards are the appropriate place to document dataset contents, license metadata, provenance, and responsible-use limitations. Public visibility on the Hub means the repository can be viewed by anyone; it does not itself establish permission to redistribute every record or override the original source's terms.

### Important source distinction

One reviewed script (`generate_labels_risk2.py`) references `sentiment_tweets3.csv` from the public GitHub project `viritaromero/Detecting-Depression-in-Tweets`. Do not automatically equate that file with the separate 10,159-tweet corpus mentioned in the manuscript. The GitHub file page reports 10,315 lines (including the header), so the exact record count and relationship to the study corpus must be established from the actual dataset version used. If the project also uses a Hugging Face mirror, cite that exact dataset ID and revision separately.

## Privacy before public release

The manuscript's statement that tweet content is not linked to author information must match the actual released files. At minimum:

1. Do not release `username`, `user_id`, `author_id`, `screen_name`, profile URLs, or equivalent account identifiers.
2. Do not release `tweet_original` or equivalent raw-post columns unless the ethics approval, source terms, and license clearly permit it. Exact post text can often be searched to recover an account even after usernames are removed.
3. Review processed text for handles, URLs, names, locations, contact details, and distinctive searchable phrases.
4. Check derived CSVs, notebooks, logs, caches, model artifacts, and filenames for identifiers or memorized text.
5. Have a human reviewer verify the release against the ethics approval/consent or waiver and applicable platform/data terms.

The helper `scripts/prepare_public_dataset.py` creates a separate candidate derivative, drops common identifier columns and raw-text columns, and applies basic pattern redaction. It is only a first-pass safeguard; it does **not** guarantee anonymization and does not authorize publication. The original input is never overwritten.

## Reproducing results without republishing tweet content

If the upstream terms or ethics approval do not permit redistribution, keep the raw dataset out of the repository. Provide the exact public source, immutable version, retrieval instructions, preprocessing code, expected schema, and permitted aggregate validation information. If access is public on Hugging Face, provide the dataset ID and revision so readers can retrieve it themselves. State explicitly if any part of the workflow cannot be rerun from public materials.
