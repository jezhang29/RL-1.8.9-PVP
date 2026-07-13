import socket
import json
import math
import torch
import torch.nn as nn

# 1. We must redefine the architecture so PyTorch knows how to load the weights
class PvPCloner(nn.Module):
    def __init__(self):
        super(PvPCloner, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.aim_head = nn.Linear(64, 2)
        self.key_head = nn.Linear(64, 7)

    def forward(self, x):
        features = self.base(x)
        return self.aim_head(features), self.key_head(features)

def calculate_optimal_angles(px, py, pz, zx, zy, zz):
    eye_x, eye_y, eye_z = px, py + 1.62, pz
    min_y, max_y = zy, zy + 1.8
    closest_y = max(min_y, min(eye_y, max_y))
    
    dx, dy, dz = zx - eye_x, closest_y - eye_y, zz - eye_z
    t_yaw = math.degrees(math.atan2(-dx, dz))
    t_pitch = math.degrees(math.atan2(-dy, math.hypot(dx, dz)))
    return t_yaw, t_pitch

def start_ai_server(host='127.0.0.1', port=9999):
    # 2. Load the trained brain
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PvPCloner().to(device)
    model.load_state_dict(torch.load('pvp_model.pth', map_location=device))
    model.eval() # Set to evaluation mode (disables training features, speeds up execution)
    print("AI Brain Loaded Successfully!")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print("Waiting for Minecraft connection...")
    conn, addr = server_socket.accept()
    print("Connected! Press 'P' in-game to unleash the AI.")

    buffer = ""
    
    try:
        with torch.no_grad(): # Don't calculate gradients during live gameplay
            while True:
                data = conn.recv(2048).decode('utf-8')
                if not data: break
                
                buffer += data
                while "\n" in buffer:
                    packet, buffer = buffer.split("\n", 1)
                    if packet.strip():
                        obs = json.loads(packet)
                        
                        action = {
                            "w": False, "a": False, "s": False, "d": False,
                            "sprint": False, "left_click": False, "right_click": False,
                            "yaw_delta": 0.0, "pitch_delta": 0.0
                        }
                        
                        dist = obs.get("target_dist", -1.0)
                        
                        if 0.0 < dist <= 6.0 and "target_x" in obs:
                            # Calculate the errors (just like we did during training)
                            t_yaw, t_pitch = calculate_optimal_angles(
                                obs["player_x"], obs["player_y"], obs["player_z"],
                                obs["target_x"], obs["target_y"], obs["target_z"]
                            )
                            
                            yaw_err = (t_yaw - obs["player_yaw"]) % 360
                            if yaw_err > 180: yaw_err -= 360
                            if yaw_err < -180: yaw_err += 360
                            pitch_err = t_pitch - obs["player_pitch"]
                            
                            # 3. Feed the current state into the Neural Network
                            # Inputs: [target_dist, yaw_error, pitch_error, on_ground]
                            x_tensor = torch.tensor([[dist, yaw_err, pitch_err, float(obs.get("on_ground", True))]], dtype=torch.float32).to(device)
                            
                            aim_preds, key_logits = model(x_tensor)
                            
                            # 4. Decode the AI's output
                            aim_deltas = aim_preds[0].cpu().numpy()
                            # Apply sigmoid to convert raw logits into probabilities (0.0 to 1.0)
                            key_probs = torch.sigmoid(key_logits[0]).cpu().numpy() 
                            
                            # If probability > 50%, press the button!
                            action["yaw_delta"] = float(aim_deltas[0])
                            action["pitch_delta"] = float(aim_deltas[1])
                            action["w"] = bool(key_probs[0] > 0.5)
                            action["a"] = bool(key_probs[1] > 0.5)
                            action["s"] = bool(key_probs[2] > 0.5)
                            action["d"] = bool(key_probs[3] > 0.5)
                            action["sprint"] = bool(key_probs[4] > 0.5)
                            action["left_click"] = bool(key_probs[5] > 0.25)
                            action["right_click"] = bool(key_probs[6] > 0.25)
                            
                        # Send the AI's decision back to Java
                        conn.sendall((json.dumps(action) + "\n").encode('utf-8'))
                        
    except KeyboardInterrupt:
        print("\nShutting down AI Server.")
    finally:
        conn.close()

if __name__ == "__main__":
    start_ai_server()