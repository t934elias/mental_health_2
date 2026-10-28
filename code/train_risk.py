
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import pickle

CSV_PATH = "./data/risk_dataset_regression.csv"
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"  
MODEL_SAVE_PATH = "./models/risk_nn_model_test.pt"
STATS_SAVE_PATH = "./models/normalization_stats_test.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH).to(DEVICE)
embed_model.eval()

def get_embedding(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return embeddings

class RiskNN(nn.Module):
    def __init__(self, input_dim):
        super(RiskNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            # nn.Dropout(p=0.15),
            # nn.Dropout(p=0.075),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            # nn.Dropout(p=0.1),
            # nn.Dropout(p=0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # nn.Dropout(p=0.05),
            # nn.Dropout(p=0.025),
            nn.Linear(32, 9)
        )
        
    def forward(self, x):
        return self.network(x)

class RiskRegressionDataset(Dataset):
    def __init__(self, dataframe, is_training=True, min_score=None, max_score=None, saved_mean=None, saved_std=None):
        if is_training:
            self.min_score = dataframe["risk_score"].min()
            self.max_score = dataframe["risk_score"].max()
        else:
            self.min_score = min_score
            self.max_score = max_score

        self.label_cols = [
            "active_nb", "passive_nb", "depression_prob", "sadness", 
            "fear", "joy", "loneliness_nb", "hopelessness_nb", "anger"
        ]

        labels = dataframe[self.label_cols].values.astype(np.float32)
        
        if is_training:
            self.label_mean = labels.mean(axis=0)
            self.label_std = labels.std(axis=0) + 1e-8
        else:
            self.label_mean = saved_mean
            self.label_std = saved_std
        
        labels = (labels - self.label_mean) / self.label_std
        self.labels = torch.tensor(labels, dtype=torch.float32)

        structured_features = dataframe[self.label_cols].values
        embeddings = np.stack([get_embedding(t) for t in dataframe["text"].values])
        
        self.features = torch.tensor(np.concatenate([embeddings, structured_features], axis=1), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    
    at_risk_df = df[df["risk_score"] > 0.05]
    zero_risk_df = df[df["risk_score"] <= 0.05]
    
    print(f"Original Data Breakdown -> At-Risk: {len(at_risk_df)}, Zero-Risk: {len(zero_risk_df)}")
    
    zero_risk_downsampled = zero_risk_df.sample(n=min(len(at_risk_df) * 2, len(zero_risk_df)), random_state=42)
    balanced_df = pd.concat([at_risk_df, zero_risk_downsampled]).reset_index(drop=True)
    print(f"Balanced Dataset Size: {len(balanced_df)}")


    train_df, temp_df = train_test_split(
        balanced_df,
        test_size=0.20,
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42
    )
    
    print(f"Training:   {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Testing:    {len(test_df)}")
    
    
    print("Processing Training Dataset...")
    train_dataset = RiskRegressionDataset(train_df, is_training=True)
    val_dataset = RiskRegressionDataset(
        val_df, 
        is_training=False, 
        min_score=train_dataset.min_score, 
        max_score=train_dataset.max_score,
        saved_mean=train_dataset.label_mean,
        saved_std=train_dataset.label_std
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    test_dataset = RiskRegressionDataset(
        test_df,
        is_training=False,
        min_score=train_dataset.min_score,
        max_score=train_dataset.max_score,
        saved_mean=train_dataset.label_mean,
        saved_std=train_dataset.label_std
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    input_dim = train_dataset.features.shape[1]
    model = RiskNN(input_dim=input_dim).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\nStarting Training Pipeline...")
    for epoch in range(30):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                val_loss += criterion(model(X_v), y_v).item()
                
        print(f"Epoch {epoch+1}/30 | Train MSE: {train_loss/len(train_loader):.4f} | Val MSE: {val_loss/len(val_loader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    meta_stats = {
        "input_dim": input_dim,
        "label_mean": train_dataset.label_mean,
        "label_std": train_dataset.label_std
    }
    with open(STATS_SAVE_PATH, "wb") as f:
        pickle.dump(meta_stats, f)
        
    print(f"\nModel and scaling parameters saved successfully.")



    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    
    print("\nEvaluating on Test Set...")
    
    model.eval()
    
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
    
        for X, y in test_loader:
    
            X = X.to(DEVICE)
    
            pred = model(X)
    
            pred = pred.cpu().numpy()
            y = y.numpy()
    
            pred = pred * train_dataset.label_std + train_dataset.label_mean
            y = y * train_dataset.label_std + train_dataset.label_mean
    
            predictions.append(pred)
            ground_truth.append(y)
    
    predictions = np.vstack(predictions)
    ground_truth = np.vstack(ground_truth)
    
    names = [
        "Active SI",
        "Passive SI",
        "Depression",
        "Sadness",
        "Fear",
        "Joy",
        "Loneliness",
        "Hopelessness",
        "Anger"
    ]
    
    print("\n===============================")
    print("Regression Performance")
    print("===============================\n")
    
    for i, name in enumerate(names):
    
        mae = mean_absolute_error(
            ground_truth[:, i],
            predictions[:, i]
        )
    
        mse = mean_squared_error(
            ground_truth[:, i],
            predictions[:, i]
        )
    
        rmse = np.sqrt(mse)
    
        r2 = r2_score(
            ground_truth[:, i],
            predictions[:, i]
        )
    
        print(f"{name}")
        print(f" MAE : {mae:.4f}")
        print(f" MSE : {mse:.4f}")
        print(f" RMSE: {rmse:.4f}")
        print(f" R²  : {r2:.4f}")
        print()


    from scipy.stats import pearsonr
    corr, _ = pearsonr(
        ground_truth[:, i],
        predictions[:, i]
    )
    
    print(f" Pearson r : {corr:.4f}")