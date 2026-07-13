import socket
import json
import math
import csv
import os

def calculate_optimal_angles(px, py, pz, zx, zy, zz, optimize_corners=False):
    # Your exact eye position
    eye_x = px
    eye_y = py + 1.62
    eye_z = pz
    
    # Target Hitbox (AABB) bounds (0.6 width/depth, 1.8 height)
    min_x, max_x = zx - 0.3, zx + 0.3
    min_y, max_y = zy, zy + 1.8
    min_z, max_z = zz - 0.3, zz + 0.3
    
    # 1. OPTIMAL PITCH: Find the closest Y on the hitbox to our eye level.
    # If on flat ground, eye_y (1.62) is between min_y (0) and max_y (1.8).
    # This clamps the target perfectly level, resulting in 0.0 pitch!
    closest_y = max(min_y, min(eye_y, max_y))
    
    # 2. OPTIMAL YAW: Center vs Corners
    if optimize_corners:
        # Mathematical absolute closest point (Hits the corners, maximizes reach)
        closest_x = max(min_x, min(eye_x, max_x))
        closest_z = max(min_z, min(eye_z, max_z))
    else:
        # Center of mass (Sacrifices tiny diagonal reach for maximum strafe tolerance)
        closest_x = zx
        closest_z = zz
        
    dx = closest_x - eye_x
    dy = closest_y - eye_y
    dz = closest_z - eye_z
    
    target_yaw = math.degrees(math.atan2(-dx, dz))
    target_pitch = math.degrees(math.atan2(-dy, math.hypot(dx, dz)))
    
    return target_yaw, target_pitch


def start_data_logger(host='127.0.0.1', port=9999):
    file_exists = os.path.isfile('pvp_dataset.csv')
    csv_file = open('pvp_dataset.csv', 'a', newline='')
    
    fieldnames = [
        'player_yaw', 'player_pitch', 'target_dist', 'ideal_yaw', 'ideal_pitch', 
        'yaw_error', 'pitch_error', 'on_ground',
        'out_w', 'out_a', 'out_s', 'out_d', 'out_sprint', 'out_left_click', 'out_right_click'
    ]
    
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print("Stage 4 Logger Online (AABB Math Active). Waiting for Minecraft...")
    conn, addr = server_socket.accept()
    print("Connected. Jump into a Duel. Press 'P' to start logging!")

    buffer = ""
    frames_recorded = 0

    try:
        while True:
            data = conn.recv(2048).decode('utf-8')
            if not data: break
            
            buffer += data
            while "\n" in buffer:
                packet, buffer = buffer.split("\n", 1)
                if packet.strip():
                    obs = json.loads(packet)
                    dist = obs.get("target_dist", -1.0)
                    
                    if 0.0 < dist <= 6.0 and "target_x" in obs:
                        # Use the new AABB clamping math
                        t_yaw, t_pitch = calculate_optimal_angles(
                            obs["player_x"], obs["player_y"], obs["player_z"],
                            obs["target_x"], obs["target_y"], obs["target_z"],
                            optimize_corners=False # Set to True if you want corner-snapping in the data!
                        )
                        
                        yaw_err = (t_yaw - obs["player_yaw"]) % 360
                        if yaw_err > 180: yaw_err -= 360
                        if yaw_err < -180: yaw_err += 360
                        pitch_err = t_pitch - obs["player_pitch"]
                        
                        writer.writerow({
                            'player_yaw': round(obs["player_yaw"], 2),
                            'player_pitch': round(obs["player_pitch"], 2),
                            'target_dist': round(dist, 3),
                            'ideal_yaw': round(t_yaw, 2),
                            'ideal_pitch': round(t_pitch, 2),
                            'yaw_error': round(yaw_err, 2),
                            'pitch_error': round(pitch_err, 2),
                            'on_ground': int(obs.get("on_ground", True)),
                            
                            'out_w': int(obs["in_w"]),
                            'out_a': int(obs["in_a"]),
                            'out_s': int(obs["in_s"]),
                            'out_d': int(obs["in_d"]),
                            'out_sprint': int(obs["in_sprint"]),
                            'out_left_click': int(obs["in_left_click"]),
                            'out_right_click': int(obs["in_right_click"])
                        })
                        
                        frames_recorded += 1
                        if frames_recorded % 100 == 0:
                            print(f"Recorded {frames_recorded} frames of active combat...")
                    
    except KeyboardInterrupt:
        print(f"\nSaved {frames_recorded} rows to pvp_dataset.csv.")
    finally:
        csv_file.close()
        conn.close()

if __name__ == "__main__":
    start_data_logger()