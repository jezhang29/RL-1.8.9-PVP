import socket
import json
from collections import deque

import torch

# Single source of truth: architecture from train_model, features from features
from train_model import PvPCloner
from features import frame_features, STACK, FRAME_DIM

# Per-key decision thresholds, calibrated on held-out data so the bot presses each
# key at the same rate the human did. A flat 0.5 doesn't work: positive-class
# weighting during training inflates the rare keys, which had the bot left-clicking
# 79% of ticks when the human clicked 35%.
#             W      A      S      D    Sprint L_Click R_Click
THRESHOLDS = [0.559, 0.360, 0.202, 0.398, 0.698, 0.543, 0.637]

def start_ai_server(host='127.0.0.1', port=9999):
    # 1. Load the trained brain
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PvPCloner().to(device)
    model.load_state_dict(torch.load('pvp_model_v2.pth', map_location=device))
    model.eval() # Evaluation mode (disables training features, speeds up execution)
    print("AI Brain Loaded Successfully!")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print("Waiting for Minecraft connection...")
    conn, addr = server_socket.accept()
    print("Connected! Press 'P' in-game to unleash the AI.")

    buffer = ""
    # Rolling window of the last STACK ticks, zero-padded until it fills
    history = deque([[0.0] * FRAME_DIM] * STACK, maxlen=STACK)
    ticks_since_click = 99  # rate-limits attacks to ~10 CPS instead of 20

    try:
        with torch.no_grad(): # Don't calculate gradients during live gameplay
            while True:
                data = conn.recv(4096).decode('utf-8')
                if not data: break

                buffer += data
                while "\n" in buffer:
                    packet, buffer = buffer.split("\n", 1)
                    if not packet.strip():
                        continue
                    obs = json.loads(packet)

                    action = {
                        "w": False, "a": False, "s": False, "d": False,
                        "sprint": False, "left_click": False, "right_click": False,
                        "yaw_delta": 0.0, "pitch_delta": 0.0
                    }

                    # 2. Build this tick's feature frame and push it into history
                    frame = frame_features(obs)
                    history.append(frame)

                    # has_target flag: only act when someone is within MAX_RANGE
                    if frame[0] == 1.0:
                        x_tensor = torch.tensor(
                            [[v for f in history for v in f]],
                            dtype=torch.float32
                        ).to(device)

                        aim_preds, key_logits = model(x_tensor)

                        # 3. Decode the AI's output
                        aim_deltas = aim_preds[0].cpu().numpy()
                        # Sigmoid converts raw logits into probabilities (0.0 to 1.0)
                        key_probs = torch.sigmoid(key_logits[0]).cpu().numpy()

                        # Safety clamp: never turn faster than a human flick in one tick
                        action["yaw_delta"] = max(-45.0, min(45.0, float(aim_deltas[0])))
                        action["pitch_delta"] = max(-30.0, min(30.0, float(aim_deltas[1])))
                        action["w"] = bool(key_probs[0] > THRESHOLDS[0])
                        action["a"] = bool(key_probs[1] > THRESHOLDS[1])
                        action["s"] = bool(key_probs[2] > THRESHOLDS[2])
                        action["d"] = bool(key_probs[3] > THRESHOLDS[3])
                        action["sprint"] = bool(key_probs[4] > THRESHOLDS[4])

                        # Attack fires a full click every tick it's true, so
                        # cap it at every other tick (~10 CPS)
                        ticks_since_click += 1
                        if key_probs[5] > THRESHOLDS[5] and ticks_since_click >= 2:
                            action["left_click"] = True
                            ticks_since_click = 0
                        action["right_click"] = bool(key_probs[6] > THRESHOLDS[6])

                    # Send the AI's decision back to Java
                    conn.sendall((json.dumps(action) + "\n").encode('utf-8'))

    except KeyboardInterrupt:
        print("\nShutting down AI Server.")
    finally:
        conn.close()

if __name__ == "__main__":
    start_ai_server()
