import csv
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import torch.nn.functional as F


import json
from ollama import chat

CSV_PATH = "./data/state_dataset_2.csv"
output_csv_path = "./data/action_label_dataset_llm_2.csv"

ACTIONS = {
    0: "empathy",
    1: "reflection",
    2: "clarifying_question",
    3: "coping_strategy",
    5: "professional_help",
    4: "safety_check",
    6: "abstain"
}

action_labels = list(ACTIONS.values())

df = pd.read_csv(CSV_PATH)

models = ["mistral:latest"]

def build_prompt(text, action_labels):
    return f"""You are a psychologist, and you want to reply to the person with the best action that will benefit them.

Choose exactly ONE intervention.

Available actions:

{action_labels}

Conversation:
{text}

Return ONLY the action name."""



with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['text', 'predicted_action_label', 'action_id', 'state_vector']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for idx, row in df.iterrows():
        text = row["text"]
        response = chat(
                model = models[0],
                messages = [
                    {
                        'role': 'user',
                        'content': build_prompt(text, action_labels),
                    },
                ]
            )
        top_label = response['message']['content'].strip().lower().replace(" ", "_")
        
        matching_ids = [k for k, v in ACTIONS.items() if v == top_label]
        action_id = matching_ids[0] if matching_ids else 6 
        
        if not matching_ids:
            top_label = ACTIONS[6]
        

        state_dict = row["user_state"]
        state_vector_string = json.dumps(state_dict)
        
        writer.writerow({
            'text': text,
            'predicted_action_label': top_label,
            'action_id': action_id,
            'state_vector': state_vector_string
        })