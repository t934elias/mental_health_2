import pandas as pd
import torch
from state_builder import build_state
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)


if __name__ == "__main__":

    dataset = pd.read_csv(
        "./data/combined_tweets.csv"
    )

    tokenizers = {
        "emotion": AutoTokenizer.from_pretrained("./models/emotion_model"),
        "risk": AutoTokenizer.from_pretrained("./models/risk_model")
    }
    
    models = {
        "emotion": AutoModelForSequenceClassification.from_pretrained("./models/emotion_model"),
        "risk": AutoModelForSequenceClassification.from_pretrained("./models/risk_model"),
        "encoder": SentenceTransformer("all-MiniLM-L6-v2")
    }


    rows = []

    for sample in dataset.to_dict('records'):

        text = sample["tweet_processed"]
        if not text or not isinstance(text, str):
            continue

        user_state = build_state(text, models, tokenizers)

        rows.append({
            "text": text, 
            "user_state": user_state, 
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv("./data/state_dataset_2.csv", index=False)
    print(df_out.head())