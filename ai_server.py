import sys
import socket
import json
from collections import deque

import numpy as np
import torch

# Single source of truth: architecture from train_model, features from features,
# combat rules from combat.
from train_model import (PvPCloner, adapt_ckpt_frame_dim, adapt_ckpt_move_dim,
                         adapt_ckpt_add_jump_head)
from features import frame_features, STACK, FRAME_DIM
from combat import CombatController, idle_action
from pvp_env import ACTIONS

#                 W      A      S      D    Sprint
MOVE_THRESHOLDS = [0.559, 0.360, 0.202, 0.398, 0.698]

# Movement to close the gap when the target is past MAX_RANGE (no training data there).
CHASE = {"w": True, "a": False, "s": False, "d": False, "sprint": True}


def _accept(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("Waiting for Minecraft connection...")
    conn, _ = server_socket.accept()
    server_socket.close()
    print("Connected! Press 'P' in-game to unleash the AI.")
    return conn


def _obs_vec(history):
    return np.array([v for f in history for v in f], dtype=np.float32)


def start_bc_bot(host='127.0.0.1', port=9999):
    """Behavioral-cloning bot: learned MOVEMENT + fully computed aim/attack (combat.act)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PvPCloner().to(device)
    model.load_state_dict(adapt_ckpt_frame_dim(
        torch.load('pvp_model_v2.pth', map_location=device)))
    model.eval()
    print("BC brain loaded (learned movement + computed aim/attack).")

    conn = _accept(host, port)
    buffer = ""
    history = deque([[0.0] * FRAME_DIM] * STACK, maxlen=STACK)
    combat = CombatController()

    try:
        with torch.no_grad():
            while True:
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break
                buffer += data
                while "\n" in buffer:
                    packet, buffer = buffer.split("\n", 1)
                    if not packet.strip():
                        continue
                    obs = json.loads(packet)
                    frame = frame_features(obs)
                    history.append(frame)

                    dist = obs.get("target_dist", -1.0)
                    if dist is None or dist <= 0.0:
                        conn.sendall((json.dumps(idle_action()) + "\n").encode('utf-8'))
                        continue

                    if frame[0] == 1.0:  # in range -> learned movement
                        x = torch.tensor([_obs_vec(history)], dtype=torch.float32).to(device)
                        key_probs = torch.sigmoid(model(x)[1][0]).cpu().numpy()
                        movement = {k: bool(key_probs[i] > MOVE_THRESHOLDS[i])
                                    for i, k in enumerate(["w", "a", "s", "d", "sprint"])}
                    else:                # past MAX_RANGE -> close the gap
                        movement = CHASE

                    action = combat.act(obs, movement)
                    conn.sendall((json.dumps(action) + "\n").encode('utf-8'))
    except KeyboardInterrupt:
        print("\nShutting down AI Server.")
    finally:
        conn.close()


def start_selfplay_bot(host='127.0.0.1', port=9999, ckpt='pvp_selfplay_best.pth'):
    """Self-play policy: learned movement AND learned attack/block/W-tap + aim residual
    (combat.act_policy). This is the trained result of train_selfplay.py."""
    # Imported here so the BC path doesn't depend on the RL module.
    from train_selfplay import ActorCritic, DEVICE

    model = ActorCritic().to(DEVICE)
    model.load_state_dict(adapt_ckpt_add_jump_head(adapt_ckpt_move_dim(
        adapt_ckpt_frame_dim(torch.load(ckpt, map_location=DEVICE)), len(ACTIONS)), model))
    model.eval()
    print(f"Self-play brain loaded from {ckpt} (learned movement + attack/block/aim).")

    conn = _accept(host, port)
    buffer = ""
    history = deque([[0.0] * FRAME_DIM] * STACK, maxlen=STACK)
    combat = CombatController()

    try:
        while True:
            data = conn.recv(4096).decode('utf-8')
            if not data:
                break
            buffer += data
            while "\n" in buffer:
                packet, buffer = buffer.split("\n", 1)
                if not packet.strip():
                    continue
                obs = json.loads(packet)
                frame = frame_features(obs)
                history.append(frame)

                dist = obs.get("target_dist", -1.0)
                if dist is None or dist <= 0.0:
                    conn.sendall((json.dumps(idle_action()) + "\n").encode('utf-8'))
                    continue

                if frame[0] == 1.0:  # in range -> the policy owns everything
                    act, _, _ = model.act(_obs_vec(history), greedy=True)
                    w, a, s, d, sprint = ACTIONS[act["move"]]
                    movement = {"w": w, "a": a, "s": s, "d": d, "sprint": sprint}
                    click, block, jump, aim = (act["click"], act["block"],
                                               act["jump"], act["aim"])
                else:                # past MAX_RANGE -> chase, still aim, don't swing
                    movement, click, block, jump, aim = CHASE, 0, 0, 0, (0.0, 0.0)

                action = combat.act_policy(obs, movement, bool(click), bool(block), aim,
                                           jump=bool(jump))
                conn.sendall((json.dumps(action) + "\n").encode('utf-8'))
    except KeyboardInterrupt:
        print("\nShutting down AI Server.")
    finally:
        conn.close()


if __name__ == "__main__":
    # `python ai_server.py`            -> behavioral-cloning bot (default)
    # `python ai_server.py selfplay`   -> trained self-play policy (pvp_selfplay_best.pth)
    if len(sys.argv) > 1 and sys.argv[1] == "selfplay":
        ckpt = sys.argv[2] if len(sys.argv) > 2 else "pvp_selfplay_best.pth"
        start_selfplay_bot(ckpt=ckpt)
    else:
        start_bc_bot()
