
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import ast
import json
import re
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import random
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.metrics import classification_report

CSV_PATH = "./data/action_label_dataset_llm_combined.csv"
SCALER_PATH = "./models/policy_scaler_llm_2.pkl"
MODEL_PATH = "./models/policy_model_llm_2.pt"



import pandas as pd

def combine_two_csvs(file1_path, file2_path, output_path):
    def safe_read_csv(filepath):
        try:
            return pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return pd.read_csv(filepath, encoding="cp1252")
            except UnicodeDecodeError:
                return pd.read_csv(filepath, encoding="utf-8", encoding_errors="replace")

    print(f"Loading {file1_path}...")
    df1 = safe_read_csv(file1_path)
    
    print(f"Loading {file2_path}...")
    df2 = safe_read_csv(file2_path)
    
    print("Combining datasets...")
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    combined_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Successfully saved combined file to: {output_path}")
    print(f"Total rows in combined dataset: {len(combined_df)}")
    


EMOTION_ORDER = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]


class MentalHealthDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.labels = torch.tensor(df["action_id"].values, dtype=torch.long)

        emotions, risks, risks_vec, uncerts, embeddings = [], [], [], [], []

        for idx, raw_state in enumerate(df["state_vector"]):

            state_dict = raw_state

            while isinstance(state_dict, str):
                state_dict = re.sub(r'np\.\w+\((.*?)\)', r'\1', state_dict)
                state_dict = re.sub(r'(?:array|tensor)\((.*?)\)', r'\1', state_dict)
                state_dict = state_dict.replace("\n", " ").replace("\r", " ")

                try:
                    state_dict = ast.literal_eval(state_dict)
                except:
                    try:
                        state_dict = json.loads(state_dict.replace("'", '"'))
                    except:
                        state_dict = {}

            emotion_vec = [
                float(state_dict.get("emotion", {}).get(k, 0.0))
                for k in EMOTION_ORDER
            ]

            risk = float(state_dict.get("risk", 0.0))
            risk_vector = state_dict.get("risk_vector", [0.0] * 9)
            risk_vector = [float(x) for x in risk_vector]

            uncert = float(state_dict.get("risk_uncertainty", 0.0))

            emb_key = "embedding" if "embedding" in state_dict else "embeddings"
            embedding = state_dict.get(emb_key, [0.0] * 384)
            embedding = [float(x) for x in embedding]

            if len(embedding) < 384:
                embedding += [0.0] * (384 - len(embedding))
            else:
                embedding = embedding[:384]

            emotions.append(emotion_vec)
            risks.append([risk])
            risks_vec.append(risk_vector)
            uncerts.append([uncert])
            embeddings.append(embedding)

        self.emotions = torch.tensor(emotions, dtype=torch.float32)
        self.risks = torch.tensor(risks, dtype=torch.float32)
        self.risks_vec = torch.tensor(risks_vec, dtype=torch.float32)
        self.uncerts = torch.tensor(uncerts, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.emotions[idx],
            self.risks[idx],
            self.risks_vec[idx],
            self.uncerts[idx],
            self.embeddings[idx],
            self.labels[idx],
        )


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.emotion_net = nn.Linear(6, 16)
        self.risk_net = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32,32)
        )
        self.risk_vec_net = nn.Sequential(
            nn.Linear(9,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )
        self.uncert_net = nn.Linear(1, 4)
        self.embed_net = nn.Linear(384, 128)

        self.classifier = nn.Sequential(
            nn.Linear(16 + 32 + 64 + 4 + 128,256),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(128,64),
            nn.ReLU(),
        
            nn.Linear(64,7)
        )

    def forward(self, emotion, risk, risk_vec, uncert, embed):
        e = torch.relu(self.emotion_net(emotion))
        r = torch.relu(self.risk_net(risk))
        rv = torch.relu(self.risk_vec_net(risk_vec))
        u = torch.relu(self.uncert_net(uncert))
        emb = torch.relu(self.embed_net(embed))

        x = torch.cat([e, r, rv, u, emb], dim=1)
        return self.classifier(x)




# FILE_1 = "./data/action_label_dataset_llm.csv"
# FILE_2 = "./data/action_label_dataset_llm_2.csv"
# OUTPUT_FILE = "./data/action_label_dataset_llm_combined.csv"

# combine_two_csvs(FILE_1, FILE_2, OUTPUT_FILE)

        


dataset = MentalHealthDataset(CSV_PATH)
labels = dataset.labels.numpy()
indices = np.arange(len(dataset))

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size


train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

temp_labels = labels[temp_idx]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=temp_labels,
    random_state=42
)

from torch.utils.data import Subset

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
test_ds = Subset(dataset, test_idx)

all_features = torch.cat([
    dataset.emotions,
    dataset.risks,
    dataset.risks_vec,
    dataset.uncerts,
    dataset.embeddings
], dim=1).numpy()

scaler = StandardScaler()
scaler.fit(all_features[train_idx])

scaled_features = scaler.transform(all_features)

e_dim = dataset.emotions.shape[1]
r_dim = dataset.risks.shape[1]
rv_dim = dataset.risks_vec.shape[1]
u_dim = dataset.uncerts.shape[1]

dataset.emotions = torch.tensor(scaled_features[:, :e_dim], dtype=torch.float32)
dataset.risks = torch.tensor(scaled_features[:, e_dim:e_dim+r_dim], dtype=torch.float32)
dataset.risks_vec = torch.tensor(scaled_features[:, e_dim+r_dim:e_dim+r_dim+rv_dim], dtype=torch.float32)
dataset.uncerts = torch.tensor(scaled_features[:, e_dim+r_dim+rv_dim:e_dim+r_dim+rv_dim+u_dim], dtype=torch.float32)
dataset.embeddings = torch.tensor(scaled_features[:, e_dim+r_dim+rv_dim+u_dim:], dtype=torch.float32)



with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)



train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



df = pd.read_csv(
    CSV_PATH
)

class_counts = (
    df["action_id"]
    .value_counts()
    .sort_index()
)


print(class_counts)

weights = 1.0 / torch.tensor(
    class_counts.values,
    dtype=torch.float32,
    device=device
)

weights = weights / weights.sum() * len(weights)


SEEDS = [0, 42, 123, 256, 512]

all_accuracies = []
all_f1 = []

for seed in SEEDS:

    print("="*60)
    print(f"Training with seed {seed}")
    print("="*60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = Policy().to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    EPOCHS = 30

    for epoch in range(EPOCHS):

        model.train()

        correct = 0
        total_samples = 0
        epoch_loss = 0

        for e,r,rv,u,emb,y in train_loader:

            e,r,rv,u,emb,y = (
                e.to(device),
                r.to(device),
                rv.to(device),
                u.to(device),
                emb.to(device),
                y.to(device)
            )

            logits = model(e,r,rv,u,emb)

            loss = criterion(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = logits.argmax(1)

            correct += (preds==y).sum().item()

            total_samples += y.size(0)

        acc = correct / total_samples

        print(
            f"Seed {seed} "
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss={epoch_loss/len(train_loader):.4f} "
            f"Acc={acc:.4f}"
        )







    model.eval()
    
    preds = []
    labels = []
    
    with torch.no_grad():
    
        for e,r,rv,u,emb,y in test_loader:
    
            e,r,rv,u,emb = (
                e.to(device),
                r.to(device),
                rv.to(device),
                u.to(device),
                emb.to(device)
            )
    
            logits = model(e,r,rv,u,emb)
    
            pred = logits.argmax(1)
    
            preds.extend(pred.cpu().numpy())
    
            labels.extend(y.numpy())
    
    acc = accuracy_score(labels,preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )
    
    all_accuracies.append(acc)
    all_f1.append(f1)
    
    print(classification_report(
        labels,
        preds,
        target_names=[
            "empathy",
            "reflection",
            "clarifying",
            "coping",
            "safety",
            "professional",
            "abstain"
        ],
        zero_division=0
    ))


print("\n")
print("="*70)
print("FINAL RESULTS OVER MULTIPLE RUNS")
print("="*70)

all_accuracies = np.array(all_accuracies)
all_f1 = np.array(all_f1)

print()

print(
    f"Accuracy: {all_accuracies.mean():.4f} ± {all_accuracies.std(ddof=1):.4f}"
)

print(
    f"Macro F1: {all_f1.mean():.4f} ± {all_f1.std(ddof=1):.4f}"
)


from scipy.stats import t

def confidence_interval(x):

    mean = np.mean(x)

    sem = stats.sem(x)

    ci = t.interval(
        confidence=0.95,
        df=len(x)-1,
        loc=mean,
        scale=sem
    )

    return ci

ci_acc = confidence_interval(all_accuracies)
ci_f1 = confidence_interval(all_f1)

print()

print("95% Confidence Interval (Accuracy):")
print(ci_acc)

print()

print("95% Confidence Interval (Macro F1):")
print(ci_f1)

