import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math

# 1. Define the Neural Network Architecture
class PvPCloner(nn.Module):
    def __init__(self):
        super(PvPCloner, self).__init__()
        
        # Base network processes the spatial state
        # Inputs: [target_dist, yaw_error, pitch_error, on_ground]
        self.base = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Head 1: Predicts mouse movement (Continuous / Regression)
        # Outputs: [yaw_delta, pitch_delta]
        self.aim_head = nn.Linear(64, 2)
        
        # Head 2: Predicts keystrokes and clicks (Binary / Classification)
        # Outputs: [W, A, S, D, Sprint, L_Click, R_Click]
        self.key_head = nn.Linear(64, 7)

    def forward(self, x):
        features = self.base(x)
        aim_deltas = self.aim_head(features)
        # We don't apply Sigmoid here because PyTorch's BCEWithLogitsLoss does it internally for better stability
        key_logits = self.key_head(features) 
        return aim_deltas, key_logits

# 2. Data Loader & Preprocessing
class PvPDataset(Dataset):
    def __init__(self, csv_file):
        print("Loading and preprocessing dataset...")
        df = pd.read_csv(csv_file)
        
        # Calculate how much the human moved the mouse per tick
        # delta = current_yaw - previous_yaw
        df['delta_yaw'] = df['player_yaw'].diff().fillna(0)
        df['delta_pitch'] = df['player_pitch'].diff().fillna(0)
        
        # Fix 360-degree wrapping (e.g., if yaw went from 359 to 1, delta is +2, not -358)
        df['delta_yaw'] = (df['delta_yaw'] + 180) % 360 - 180
        
        # Filter out massive camera snaps (likely caused by toggling the logger on/off)
        df = df[(df['delta_yaw'].abs() < 90) & (df['delta_pitch'].abs() < 90)]
        
        # Inputs (X)
        self.X = df[['target_dist', 'yaw_error', 'pitch_error', 'on_ground']].values.astype(np.float32)
        
        # Outputs (Y) - split into Aim and Keys
        self.Y_aim = df[['delta_yaw', 'delta_pitch']].values.astype(np.float32)
        self.Y_keys = df[['out_w', 'out_a', 'out_s', 'out_d', 'out_sprint', 'out_left_click', 'out_right_click']].values.astype(np.float32)
        
        print(f"Dataset ready: {len(self.X)} valid frames.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_aim[idx], self.Y_keys[idx]

# 3. Training Loop
def train():
    dataset = PvPDataset('pvp_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = PvPCloner().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Two different loss functions for the two different heads
    criterion_aim = nn.MSELoss() # Mean Squared Error for mouse smoothing
    criterion_keys = nn.BCEWithLogitsLoss() # Binary Cross Entropy for key presses
    
    epochs = 30
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_X, batch_Y_aim, batch_Y_keys in dataloader:
            batch_X = batch_X.to(device)
            batch_Y_aim = batch_Y_aim.to(device)
            batch_Y_keys = batch_Y_keys.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_aim, pred_keys = model(batch_X)
            
            # Calculate combined loss (Aiming is harder to learn, so we weight it heavily)
            loss_aim = criterion_aim(pred_aim, batch_Y_aim)
            loss_keys = criterion_keys(pred_keys, batch_Y_keys)
            loss = loss_aim + (loss_keys * 10.0) 
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(dataloader):.4f}")
        
    print("Training complete! Saving model weights...")
    torch.save(model.state_dict(), 'pvp_model.pth')
    print("Saved as 'pvp_model.pth'.")

if __name__ == "__main__":
    train()