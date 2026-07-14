import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from features import frame_features, wrap_degrees, STACK, INPUT_DIM

# 1. Define the Neural Network Architecture
class PvPCloner(nn.Module):
    def __init__(self):
        super(PvPCloner, self).__init__()

        # Base network processes the last STACK ticks of state
        self.base = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Head 1: Predicts mouse movement (Continuous / Regression)
        # Outputs: [yaw_delta, pitch_delta]
        self.aim_head = nn.Linear(128, 2)

        # Head 2: Predicts keystrokes and clicks (Binary / Classification)
        # Outputs: [W, A, S, D, Sprint, L_Click, R_Click]
        self.key_head = nn.Linear(128, 7)

    def forward(self, x):
        features = self.base(x)
        aim_deltas = self.aim_head(features)
        # No Sigmoid here - BCEWithLogitsLoss applies it internally for stability
        key_logits = self.key_head(features)
        return aim_deltas, key_logits

# 2. Data Loader & Preprocessing
class PvPDataset(Dataset):
    def __init__(self, csv_file):
        print("Loading and preprocessing dataset...")
        df = pd.read_csv(csv_file)
        rows = df.to_dict('records')

        # Recordings made before the mod logged mc.thePlayer.isSprinting() read the
        # sprint *keybind*, which double-tap-W sprinting never presses - so sprint came
        # out all-zero. Recover it from speed: the 1.8.9 walk cap is ~0.215 blocks/tick,
        # sprint tops out near 0.28, so anything above 0.235 was a sprint.
        if df['out_sprint'].mean() < 0.01:
            speed = np.hypot(df['my_vx'], df['my_vz'])
            df['out_sprint'] = (speed > 0.235).astype(int)
            print(f"  ! out_sprint was empty - recovered {df['out_sprint'].mean():.1%} "
                  f"of frames as sprinting, inferred from movement speed.")

        frames = np.array([frame_features(r) for r in rows], dtype=np.float32)
        ticks = df['tick'].values
        yaws = df['player_yaw'].values
        pitches = df['player_pitch'].values
        keys = df[['out_w', 'out_a', 'out_s', 'out_d', 'out_sprint',
                   'out_left_click', 'out_right_click']].values.astype(np.float32)

        # contiguous[i] is True when row i+1 is the very next game tick,
        # so stacks and labels never bridge a recording gap or new session
        contiguous = np.diff(ticks) == 1

        X, Y_aim, Y_keys = [], [], []
        for t in range(STACK - 1, len(rows) - 1):
            # Need every step from the start of the stack through t+1 (the label)
            if not contiguous[t - STACK + 1 : t + 1].all():
                continue
            # Only learn from moments where a target is actually in range
            if frames[t][0] != 1.0:
                continue

            # The move made in RESPONSE to the state at t lands in row t+1
            delta_yaw = wrap_degrees(yaws[t + 1] - yaws[t])
            delta_pitch = pitches[t + 1] - pitches[t]
            if abs(delta_yaw) > 90 or abs(delta_pitch) > 90:
                continue

            X.append(frames[t - STACK + 1 : t + 1].flatten())
            Y_aim.append([delta_yaw, delta_pitch])
            Y_keys.append(keys[t])

        self.X = np.array(X, dtype=np.float32)
        self.Y_aim = np.array(Y_aim, dtype=np.float32)
        self.Y_keys = np.array(Y_keys, dtype=np.float32)

        print(f"Dataset ready: {len(self.X)} valid frames.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_aim[idx], self.Y_keys[idx]

# 3. Training Loop
KEY_NAMES = ['W', 'A', 'S', 'D', 'Sprint', 'L_Click', 'R_Click']

def train():
    dataset = PvPDataset('pvp_dataset_v2.csv')

    # Split by time, not at random: consecutive ticks are nearly identical, so a
    # random split leaks almost every val sample into the train set and the val
    # score comes out meaninglessly high.
    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_set = torch.utils.data.Subset(dataset, range(n_train))
    val_set = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))
    print(f"Train: {n_train} frames | Validation (held-out tail): {n_val} frames")

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = PvPCloner().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    epochs = 80
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Rare actions (blockhitting, S-tapping) get drowned out by the 96% of frames
    # where they're off, so the net learns "never press it". Weight the positives up
    # by how rare they are - capped, or a 0.8%-frequency key would get a 120x boost
    # and the bot would mash it.
    freq = dataset.Y_keys[:n_train].mean(axis=0).clip(0.01, 0.99)
    pos_weight = torch.tensor(((1 - freq) / freq).clip(1.0, 5.0), dtype=torch.float32).to(device)
    print("Key frequencies: " + ", ".join(f"{n}={f:.1%}" for n, f in zip(KEY_NAMES, freq)))

    criterion_aim = nn.MSELoss()
    criterion_keys = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_Y_aim, batch_Y_keys in train_loader:
            batch_X, batch_Y_aim, batch_Y_keys = (
                batch_X.to(device), batch_Y_aim.to(device), batch_Y_keys.to(device))

            optimizer.zero_grad()
            pred_aim, pred_keys = model(batch_X)
            loss = criterion_aim(pred_aim, batch_Y_aim) + 10.0 * criterion_keys(pred_keys, batch_Y_keys)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_aim_err, correct, n = 0, 0, np.zeros(7), 0
        with torch.no_grad():
            for batch_X, batch_Y_aim, batch_Y_keys in val_loader:
                batch_X, batch_Y_aim, batch_Y_keys = (
                    batch_X.to(device), batch_Y_aim.to(device), batch_Y_keys.to(device))
                pred_aim, pred_keys = model(batch_X)
                val_loss += (criterion_aim(pred_aim, batch_Y_aim)
                             + 10.0 * criterion_keys(pred_keys, batch_Y_keys)).item()
                val_aim_err += (pred_aim - batch_Y_aim).abs().mean(dim=0)[0].item() * len(batch_X)
                hits = ((torch.sigmoid(pred_keys) > 0.5) == (batch_Y_keys > 0.5))
                correct += hits.sum(dim=0).cpu().numpy()
                n += len(batch_X)

        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'pvp_model_v2.pth')
            tag = " <- best, saved"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == epochs - 1 or tag:
            print(f"Epoch [{epoch+1}/{epochs}] train {total_loss/len(train_loader):.4f} | "
                  f"val {val_loss:.4f} | yaw err {val_aim_err/n:.2f} deg{tag}")

    print(f"\nBest validation loss: {best_val:.4f}")
    print("Per-key accuracy on held-out data:")
    for name, acc in zip(KEY_NAMES, correct / n):
        print(f"  {name:8s} {acc:.1%}")
    print("\nSaved best weights to 'pvp_model_v2.pth'.")

if __name__ == "__main__":
    train()
