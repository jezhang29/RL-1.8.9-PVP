import math

# Shared feature engineering for train_model.py and ai_server.py.
# Both import from here so training and inference can never drift apart.

STACK = 8          # ticks of history fed to the network (0.4s at 20Hz)
FRAME_DIM = 27     # features per tick, see frame_features()
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

    def clamp(v, lim=3.0):
        # A network only ever saw roughly [-1, 1] here. Letting a live feature run
        # to -2.8 (which is what a crosshair buried in the ground does to pitch_err)
        # puts it deep in extrapolation territory, where it emits nonsense - phantom
        # S-key, random sprinting. Clamping bounds the damage from a bad state.
        return max(-lim, min(lim, v))

    # --- v3 appended block: ping / opponent intent / wall context. Every one of
    # these reads 0 from old jars and old CSVs (absent key or -1 sentinel), and 0
    # must mean "neutral / LAN / nothing there" so remapped old checkpoints keep
    # computing exactly what they did before.
    def ping01(key):
        # Tab-list round-trip in ms -> 1.0 at 200ms; -1/absent (LAN, old jar) -> 0
        p = obs.get(key)
        return clamp(max(p or 0.0, 0.0) / 200.0)

    # How far the target is looking AWAY from us: 0 = staring at us, 1 = back
    # turned. Absent (old jar/CSV) -> 0 = the worst-case "they see us" assumption.
    tgt_yaw_raw = obs.get("tgt_yaw")
    if tgt_yaw_raw is None:
        facing_away = 0.0
    else:
        bearing_tgt_to_me = math.degrees(math.atan2(
            -(obs["player_x"] - obs["target_x"]),
            obs["player_z"] - obs["target_z"]))
        facing_away = abs(wrap_degrees(bearing_tgt_to_me - tgt_yaw_raw)) / 180.0

    def wall01(key):
        # Mod sends blocks-to-first-solid along the knockback lane, in (0, 3];
        # -1/absent = open. Inverted to "wall pressure" so 0 stays neutral: 0.83
        # pinned against the wall, 0 nothing within 3 blocks.
        d = obs.get(key)
        if d is None or d <= 0.0:
            return 0.0
        return 1.0 - min(d, 3.0) / 3.0

    return [
        1.0,                                                   # has_target
        dist / MAX_RANGE,
        clamp(yaw_err / 30.0),
        clamp(pitch_err / 30.0),
        clamp((obs["target_y"] - obs["player_y"]) / 4.0),      # height diff (knockback arcs)
        float(obs["on_ground"]),
        clamp(my_vf / 0.3), clamp(my_vs / 0.3),                # ~sprint speed = 0.28 blocks/tick
        clamp(tg_vf / 0.3), clamp(tg_vs / 0.3),
        obs["my_hurt"] / 10.0,                                 # hurtTime counts down 10 -> 0
        obs["tgt_hurt"] / 10.0,                                # THE combo-timing signal
        # Sprint/block state, both fighters (needs the updated jar; absent keys -> 0,
        # so old CSVs and old jars keep working). Sprint = the knockback-bonus flag
        # we manage with W-taps; block = using-item, i.e. moving at 20% speed and
        # taking half damage. Appended at the END of the frame so checkpoints trained
        # on the 12-feature layout can be column-remapped (adapt_ckpt_frame_dim).
        float(obs.get("my_sprint") or 0.0),
        float(obs.get("tgt_sprint") or 0.0),
        float(obs.get("my_block") or 0.0),
        float(obs.get("tgt_block") or 0.0),
        # v3 appended block (see helpers above); append-only, per convention
        ping01("my_ping"),
        ping01("tgt_ping"),
        float(obs.get("tgt_swing") or 0.0),                    # they're mid-attack NOW
        facing_away,
        wall01("my_wall"),                                     # am I getting cornered
        wall01("tgt_wall"),                                    # are THEY cornered
        # v4 appended block: vertical velocity, target air state, mutual ray checks.
        # vy/0.4: jump launch ~ +1, knockback arc peak ~ +1 (base KB caps vy at 0.4)
        clamp((obs.get("my_vy") or 0.0) / 0.4),
        clamp((obs.get("tgt_vy") or 0.0) / 0.4),
        # Airborne = mid-knockback/juggleable (air friction 0.91 vs ~0.55 grounded).
        # Inverted from on_ground so absent (old jar/CSV) -> 0 = grounded = neutral.
        0.0 if obs.get("tgt_on_ground") is None else 1.0 - float(obs["tgt_on_ground"]),
        float(obs.get("my_ray_hit") or 0.0),                   # a click right now connects
        float(obs.get("tgt_ray_hit") or 0.0),                  # THEIR next click connects
    ]
