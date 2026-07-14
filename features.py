import math

# Shared feature engineering for train_model.py and ai_server.py.
# Both import from here so training and inference can never drift apart.

STACK = 8          # ticks of history fed to the network (0.4s at 20Hz)
FRAME_DIM = 12     # features per tick, see frame_features()
INPUT_DIM = STACK * FRAME_DIM
MAX_RANGE = 16.0   # blocks; beyond this the bot has no data and idles

def calculate_optimal_angles(px, py, pz, zx, zy, zz):
    eye_x, eye_y, eye_z = px, py + 1.62, pz
    # Clamp eye level onto the target's hitbox height for optimal pitch
    closest_y = max(zy, min(eye_y, zy + 1.8))
    dx, dy, dz = zx - eye_x, closest_y - eye_y, zz - eye_z
    t_yaw = math.degrees(math.atan2(-dx, dz))
    t_pitch = math.degrees(math.atan2(-dy, math.hypot(dx, dz)))
    return t_yaw, t_pitch

def wrap_degrees(angle):
    angle = angle % 360
    if angle > 180: angle -= 360
    return angle

def frame_features(obs):
    """One FRAME_DIM-float feature frame from a raw telemetry dict.

    Works on both live JSON from Java and a row of the v2 CSV.
    Returns all zeros (has_target=0) when no target is in range, so
    'target ran away mid-stack' looks identical in training and live play.
    """
    dist = obs.get("target_dist", -1.0)
    if dist is None or not (0.0 < dist <= MAX_RANGE):
        return [0.0] * FRAME_DIM

    t_yaw, t_pitch = calculate_optimal_angles(
        obs["player_x"], obs["player_y"], obs["player_z"],
        obs["target_x"], obs["target_y"], obs["target_z"]
    )
    yaw_err = wrap_degrees(t_yaw - obs["player_yaw"])
    pitch_err = t_pitch - obs["player_pitch"]

    # Rotate world-space velocities into the player's view frame so the
    # model sees "target moving left/right/toward/away", not compass axes
    yaw_rad = math.radians(obs["player_yaw"])
    s, c = math.sin(yaw_rad), math.cos(yaw_rad)
    def to_local(vx, vz):
        return (-vx * s + vz * c,   # forward component
                 vx * c + vz * s)   # strafe component
    my_vf, my_vs = to_local(obs["my_vx"], obs["my_vz"])
    tg_vf, tg_vs = to_local(obs["tgt_vx"], obs["tgt_vz"])

    return [
        1.0,                                            # has_target
        dist / MAX_RANGE,
        yaw_err / 30.0,
        pitch_err / 30.0,
        (obs["target_y"] - obs["player_y"]) / 4.0,      # height diff (knockback arcs)
        float(obs["on_ground"]),
        my_vf / 0.3, my_vs / 0.3,                       # ~sprint speed = 0.28 blocks/tick
        tg_vf / 0.3, tg_vs / 0.3,
        obs["my_hurt"] / 10.0,                          # hurtTime counts down 10 -> 0
        obs["tgt_hurt"] / 10.0,                         # THE combo-timing signal
    ]
