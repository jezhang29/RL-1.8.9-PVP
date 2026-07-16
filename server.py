import socket
import json
import csv
import os

# v2 Data Logger.
# Logs RAW state every tick while recording (toggle with 'O' in-game) -
# feature engineering happens later in features.py so we can iterate on
# features without re-recording. Frames without a target are logged too,
# so frame stacks stay contiguous when someone briefly runs out of range.

# v3 adds my_vy / my_health / tgt_health / out_jump, so it gets its own file -
# appending these rows to the v2 CSV would silently shift every column.
CSV_PATH = 'pvp_dataset_v3.csv'

FIELDNAMES = [
    'tick',
    'player_x', 'player_y', 'player_z', 'player_yaw', 'player_pitch',
    'on_ground', 'my_vx', 'my_vy', 'my_vz', 'my_hurt', 'my_health',
    'target_x', 'target_y', 'target_z', 'target_dist',
    'tgt_vx', 'tgt_vz', 'tgt_hurt', 'tgt_health',
    'out_w', 'out_a', 'out_s', 'out_d', 'out_sprint', 'out_jump',
    'out_left_click', 'out_right_click'
]

def start_data_logger(host='127.0.0.1', port=9999):
    # an existing-but-empty file still needs a header
    file_exists = os.path.isfile(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    csv_file = open(CSV_PATH, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print("v2 Logger Online. Waiting for Minecraft...")
    conn, addr = server_socket.accept()
    print("Connected. Jump into a Duel. Press 'O' in-game to toggle recording!")

    buffer = ""
    frames_recorded = 0

    try:
        while True:
            data = conn.recv(4096).decode('utf-8')
            if not data: break

            buffer += data
            while "\n" in buffer:
                packet, buffer = buffer.split("\n", 1)
                if not packet.strip():
                    continue
                obs = json.loads(packet)

                if not obs.get("recording", False):
                    continue

                has_target = "target_x" in obs
                writer.writerow({
                    'tick': obs["tick"],
                    'player_x': obs["player_x"],
                    'player_y': obs["player_y"],
                    'player_z': obs["player_z"],
                    'player_yaw': round(obs["player_yaw"], 3),
                    'player_pitch': round(obs["player_pitch"], 3),
                    'on_ground': int(obs.get("on_ground", True)),
                    'my_vx': round(obs["my_vx"], 4),
                    'my_vy': round(obs["my_vy"], 4),
                    'my_vz': round(obs["my_vz"], 4),
                    'my_hurt': obs["my_hurt"],
                    'my_health': round(obs["my_health"], 2),
                    'target_x': obs["target_x"] if has_target else '',
                    'target_y': obs["target_y"] if has_target else '',
                    'target_z': obs["target_z"] if has_target else '',
                    'target_dist': round(obs["target_dist"], 3) if has_target else '',
                    'tgt_vx': round(obs["tgt_vx"], 4) if has_target else '',
                    'tgt_vz': round(obs["tgt_vz"], 4) if has_target else '',
                    'tgt_hurt': obs["tgt_hurt"] if has_target else '',
                    'tgt_health': round(obs["tgt_health"], 2) if has_target else '',

                    'out_w': int(obs["in_w"]),
                    'out_a': int(obs["in_a"]),
                    'out_s': int(obs["in_s"]),
                    'out_d': int(obs["in_d"]),
                    'out_sprint': int(obs["in_sprint"]),
                    'out_jump': int(obs["in_jump"]),
                    'out_left_click': int(obs["in_left_click"]),
                    'out_right_click': int(obs["in_right_click"])
                })

                frames_recorded += 1
                if frames_recorded % 100 == 0:
                    csv_file.flush()
                    print(f"Recorded {frames_recorded} frames...")

    except KeyboardInterrupt:
        print(f"\nSaved {frames_recorded} rows to {CSV_PATH}.")
    finally:
        csv_file.close()
        conn.close()

if __name__ == "__main__":
    start_data_logger()
