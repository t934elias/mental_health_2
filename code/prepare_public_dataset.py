#!/usr/bin/env python3
"""Create a first-pass privacy-reviewed CSV derivative.

This is NOT anonymization and does not grant permission to redistribute data.
It never overwrites the input. A human ethics/privacy review is mandatory.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd

IDENTIFIER_COLUMN = re.compile(
    r"(^|[_\W])(username|user_name|user_id|userid|author|account|profile|"
    r"screen_name|handle|tweet_id|author_id|profile_url)([_\W]|$)", re.I
)
RAW_TEXT_COLUMNS = {
    "tweet_original", "original_tweet", "raw_tweet", "tweet_text_raw", "text_original"
}
HANDLE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")


def redact_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    value = URL.sub("[URL]", value)
    value = EMAIL.sub("[EMAIL]", value)
    value = HANDLE.sub("[USER_HANDLE]", value)
    return PHONE.sub("[PHONE]", value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--text-columns", nargs="*", default=["tweet_processed"],
        help="Columns to apply basic pattern redaction to (default: tweet_processed).",
    )
    args = parser.parse_args()
    src, dst = args.input_csv.resolve(), args.output_csv.resolve()
    if not src.is_file():
        parser.error(f"Input file does not exist: {src}")
    if src == dst:
        parser.error("Input and output must differ; refusing to overwrite source.")
    if dst.exists():
        parser.error(f"Output already exists; refusing to overwrite: {dst}")

    df = pd.read_csv(src)
    n_rows = len(df)
    drop = [c for c in df.columns if IDENTIFIER_COLUMN.search(str(c))
            or str(c).strip().lower() in RAW_TEXT_COLUMNS]
    df = df.drop(columns=drop, errors="ignore")
    redacted = []
    for col in args.text_columns:
        if col in df.columns:
            df[col] = df[col].map(redact_text)
            redacted.append(col)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"Rows: {n_rows} -> {len(df)}")
    print(f"Dropped columns: {drop}")
    print(f"Pattern-redacted columns: {redacted}")
    print(f"Wrote: {dst}")
    print("WARNING: not anonymized; inspect manually and obtain ethics/legal approval.")


if __name__ == "__main__":
    main()
