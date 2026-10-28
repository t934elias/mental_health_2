
import torch
import torch.nn as nn
import numpy as np
from state_builder import build_state
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


with open("./models/policy_scaler_llm_2.pkl", "rb") as f:
    scaler = pickle.load(f)

EMOTION_ORDER = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]


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

    def forward(self, e, r, rv, u, emb):
        e = torch.relu(self.emotion_net(e))
        r = torch.relu(self.risk_net(r))
        rv = torch.relu(self.risk_vec_net(rv))
        u = torch.relu(self.uncert_net(u))
        emb = torch.relu(self.embed_net(emb))

        x = torch.cat([e, r, rv, u, emb], dim=1)
        return self.classifier(x)


ACTIONS = {
    0: "empathy",
    1: "reflection",
    2: "clarifying_question",
    3: "coping_strategy",
    4: "safety_check",
    5: "professional_help",
    6: "abstain"
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Policy().to(device)
model.load_state_dict(torch.load("./models/policy_model_llm_2.pt", map_location=device))
model.eval()


tokenizers = {
    "emotion": AutoTokenizer.from_pretrained("./models/emotion_model"),
    "risk": AutoTokenizer.from_pretrained("./models/risk_model")
}

models = {
    "emotion": AutoModelForSequenceClassification.from_pretrained("./models/emotion_model"),
    "risk": AutoModelForSequenceClassification.from_pretrained("./models/risk_model"),
    "encoder": SentenceTransformer("all-MiniLM-L6-v2")
}

CSV_PATH = "./data/action_label_dataset_llm_2.csv"


dataset = [
    "i am happy",
    "i want to die",
    "why is life so hard",
    "I don't want to live anymore.",
    "I feel so lonely and isolated from everyone, nothing matters anymore."
]


for text in dataset:

    user_state = build_state(text, models, tokenizers)
    
    emotion = np.array([
        float(user_state["emotion"].get(k, 0.0))
        for k in EMOTION_ORDER
    ]).reshape(1, -1)

    risk = np.array([[float(user_state.get("risk", 0.0))]])
    risk_vec = np.array(user_state["risk_vector"]).reshape(1, -1)
    uncert = np.array([[float(user_state.get("risk_uncertainty", 0.0))]])

    embedding = np.array(user_state["embedding"]).reshape(1, -1)

    features = np.concatenate([
        emotion,
        risk,
        risk_vec,
        uncert,
        embedding
    ], axis=1)

    features = scaler.transform(features)
    
    e_dim = emotion.shape[1]      # 6
    r_dim = risk.shape[1]         # 1
    rv_dim = risk_vec.shape[1]    # 9
    u_dim = uncert.shape[1]       # 1

    emotion = features[:, :e_dim]
    risk = features[:, e_dim:e_dim+r_dim]
    risk_vec = features[:, e_dim+r_dim:e_dim+r_dim+rv_dim]
    uncert = features[:, e_dim+r_dim+rv_dim:e_dim+r_dim+rv_dim+u_dim]
    embedding = features[:, e_dim+r_dim+rv_dim+u_dim:]

    e = torch.tensor(emotion, dtype=torch.float32).to(device)
    r = torch.tensor(risk, dtype=torch.float32).to(device)
    rv = torch.tensor(risk_vec, dtype=torch.float32).to(device)
    u = torch.tensor(uncert, dtype=torch.float32).to(device)
    emb = torch.tensor(embedding, dtype=torch.float32).to(device)


    with torch.no_grad():
        logits = model(e, r, rv, u, emb)
        probs = torch.softmax(logits, dim=1)[0]

        pred = torch.argmax(probs).item()

    print("\n========================")
    print(text)
    print("Risk:", user_state["risk"])
    print("Prediction:", ACTIONS[pred])
    print("Probs:")

    for i, name in ACTIONS.items():
        print(f"{name:<25} {probs[i].item():.4f}")

    topk = torch.topk(probs, k=3)

    print("\nTop predictions:")
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        print(f"{ACTIONS[idx]:<25} {p:.3f}")

