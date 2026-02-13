import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================================================
# ⚙️ CONFIGURATION (The "Hyperparameters")
# =========================================================
# Window Size: How far back does the AI look? (e.g., 10 previous cycles)
SEQUENCE_LENGTH = 30         
BATCH_SIZE = 32
EPOCHS = 50  # How many times to loop through the entire dataset
LEARNING_RATE = 0.0001

# Paths (Auto-detected)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'data' / 'models'

# =========================================================
# 🧠 MODEL ARCHITECTURE (The "Brain")
# =========================================================
class ScrapLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1):
        super(ScrapLSTM, self).__init__()
        
        # LSTM Layer: Captures time-dependent patterns (Trends)
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        # Fully Connected Layer: Makes the final Yes/No decision
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # Squashes output to 0-1 probability

    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        
        # Forward propagate LSTM
        # out shape: [batch_size, seq_len, hidden_dim]
        out, (hn, cn) = self.lstm(x)
        
        # We only care about the result of the LAST time step
        out = out[:, -1, :] 
        
        # Pass through linear layer
        out = self.fc(out)
        return self.sigmoid(out)

# =========================================================
# 📊 DATA PREPARATION (The "Movie Maker")
# =========================================================
class ManufacturingDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_sequences(data, target, seq_length):
    """
    Converts a flat table into 3D sequences for LSTM.
    Input: [Row1, Row2, Row3...]
    Output: [[Row1-10], [Row2-11], [Row3-12]...]
    """
    xs, ys = [], []
    data_array = data.values
    target_array = target.values

    for i in range(len(data) - seq_length):
        x = data_array[i:(i + seq_length)]
        y = target_array[i + seq_length]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

# =========================================================
# 🚀 MAIN TRAINING LOOP
# =========================================================
def train_deep_learning():
    print("============================================================")
    print("🚀 STARTING DEEP LEARNING TRAINING (LSTM MODE)")
    print("============================================================")

    # 1. Load Data (We use the clean data from the previous pipeline)
    data_path = DATA_DIR / "processed_full_dataset.pkl"
    if not data_path.exists():
        print("❌ Error: Processed data not found. Run 'run_pipeline.py' first.")
        return

    print("📥 Loading dataset...")
    df = pd.read_pickle(data_path)
    
    # 2. Separate Features & Target (CORRECTED)
    # ==========================================
    # 2. SEPARATE FEATURES & TARGET (CRASH-PROOF)
    # ==========================================
    
    # Step A: Select ONLY numbers (Fixes the Date error)
    X = df.select_dtypes(include=[np.number])
    
    # Step B: Remove the Answer Key ('is_scrap') from the Input
    if 'is_scrap' in X.columns:
        X = X.drop(columns=['is_scrap'])
        
    # Step C: CRITICAL FIX - Fill Empty/Infinite Values
    # This fixes "All-NaN slice" and "RuntimeError"
    X = X.fillna(0.0)
    X = X.replace([np.inf, -np.inf], 0.0)
    
    # Check if we accidentally deleted everything
    if X.empty:
        print("❌ Error: No valid numeric columns found! Check your data.")
        return

    y = df['is_scrap']
    
    # Update feature list for tracking
    feature_cols = X.columns.tolist()
    
    print(f"✅ Data Cleaned. Features: {len(feature_cols)} | Samples: {len(X)}")

    # 3. Scaling (CRITICAL for Deep Learning)
    # LSTMs fail if data isn't between 0 and 1 or -1 and 1
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Save scaler for later use
    joblib.dump(scaler, MODELS_DIR / "deep_scaler.joblib")

    # 4. Create Sequences (The "Movie" View)
    print(f"🎞️  Creating time sequences (Window: {SEQUENCE_LENGTH})...")
    X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=True)
    
    # Create DataLoaders
    train_dataset = ManufacturingDataset(X_train, y_train)
    test_dataset = ManufacturingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) # Order matters in time series!
    
    # 5. Initialize Model
    model = ScrapLSTM(input_dim=len(feature_cols))
    criterion = nn.BCELoss() # Binary Cross Entropy (Standard for Yes/No)
    # weight_decay=1e-5 prevents the model from overfitting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # 6. Training Loop
    print("\n🧠 Training Neural Network...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    # 7. Evaluation
    print("\n📝 Evaluating Performance...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        predictions = model(X_test_tensor).squeeze()
        preds_binary = (predictions > 0.5).float()
        
        acc = accuracy_score(y_test_tensor, preds_binary)
        auc = roc_auc_score(y_test_tensor, predictions)
        
        print(f"✅ Accuracy: {acc:.2%}")
        print(f"✅ ROC-AUC:  {auc:.4f}")

    # 8. Save the Model
    torch.save(model.state_dict(), MODELS_DIR / "lstm_scrap_model.pth")
    print(f"\n💾 Model saved to: {MODELS_DIR / 'lstm_scrap_model.pth'}")
    print("🎉 Deep Learning Pipeline Complete!")

if __name__ == "__main__":
    train_deep_learning()