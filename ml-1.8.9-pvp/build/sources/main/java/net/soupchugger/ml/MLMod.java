package net.soupchugger.ml;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGameOver;
import net.minecraft.client.network.NetworkPlayerInfo;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.BlockPos;
import net.minecraft.util.ChatComponentText;
import net.minecraft.util.Vec3;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.client.registry.ClientRegistry;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;
import org.lwjgl.input.Keyboard;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

@Mod(
        modid   = MLMod.MODID,
        name    = MLMod.NAME,
        version = MLMod.VERSION,
        clientSideOnly = true
)
public class MLMod {

    public static final String MODID   = "mlpvp";
    public static final String NAME    = "ML PVP";
    public static final String VERSION = "1.0.0";

    private final Minecraft mc = Minecraft.getMinecraft();
    private PrintWriter out;
    private BufferedReader in;
    private boolean isConnecting = false;

    // Python backend port. Both self-play clients run this same jar, so the SECOND
    // client must override this to 9998 with a JVM arg: -Dmlpvp.port=9998 (the learner
    // stays on the default 9999). Single-client use (BC bot / recording) needs nothing.
    private final int aiPort = Integer.getInteger("mlpvp.port", 9999);

    // 1. Create the Keybindings and State Trackers
    private KeyBinding aiToggleKey;     // P - AI takes over the controls
    private KeyBinding recordToggleKey; // O - stream "recording=true" so the logger saves frames
    private boolean isAiActive = false;
    private boolean isRecording = false;
    private long tickCounter = 0; // lets Python detect session gaps when stacking frames

    // Model now predicts true per-tick deltas (label off-by-one fixed in training),
    // so no artificial boost is needed
    private float aiAimSensitivity = 1.0f;

    // Camera smoothing. We apply the whole per-tick rotation at tick START (physics
    // reads the final angle, so hit registration is unchanged). But the entity's
    // onUpdate() copies prevRotation = rotation during the tick, which zeroes the
    // camera's frame-to-frame interpolation and makes AI aim look 20 fps / steppy. At
    // tick END we restore prevRotation to the PRE-delta angle so the vanilla camera
    // sweeps prev->current smoothly across every render frame at the real framerate.
    private float preYaw, prePitch;
    private boolean rotationSmoothPending = false;

    // The "Mailbox" - Volatile ensures thread safety when reading/writing
    private volatile BotAction currentAction = new BotAction();
    // When the last action arrived. Movement/sprint/block are HELD state re-applied
    // every tick, so if Python pauses (a PPO update runs for seconds; each round
    // reset starves the other client ~1s) the fighter would keep replaying its last
    // action forever - walking into a wall holding block until the pause ends. A
    // stale action isn't a decision: past STALE_ACTION_MS, idle instead.
    private static final long STALE_ACTION_MS = 500;   // ~10 ticks
    private volatile long lastActionMillis = 0;
    // A chat command the AI wants run (e.g. "/tp" to reset a self-play round).
    // Must be executed on the client thread, so onClientTick drains it.
    private volatile String pendingCommand = null;

    // Control-loop staleness. Python echoes back (ack_tick) the tick of the state
    // each action was computed from. Healthy pipelining is staleness 1 at apply time:
    // this tick we apply the reply to LAST tick's state. When the two clients' tick
    // clocks sit nearly a full cycle apart, Python (which replies only after hearing
    // from BOTH fighters) answers one client just AFTER its next tick starts - that
    // client then applies every action a tick late (staleness 2). Double control
    // latency at the same AIM_GAIN is what reads as the crosshair orbiting/"spinning",
    // and because both tick clocks are periodic the bad phase persists all session
    // (any pause, e.g. opening a GUI, re-rolls it - which is why toggling video
    // settings "fixed" it). Fix: when the reply to last tick's state hasn't arrived
    // yet, wait up to FRESH_WAIT_MS for it instead of applying the stale action.
    // Costs 0 in a good phase, a few ms/tick in a bad one. Skipped when Python is
    // genuinely paused (lag > 4 ticks, e.g. a PPO update) or never acks (BC server).
    private static final int FRESH_WAIT_MS = 25;
    private long lastStaleness = -1;                       // reported in telemetry
    private long staleSum = 0, staleMax = 0, staleTicks = 0, staleWaitedMs = 0;

    // Data container for our inputs
    public static class BotAction {
        public boolean w, a, s, d, sprint, jump;
        public float yaw_delta, pitch_delta;
        public boolean left_click, right_click;
        public long ack_tick = -1;   // tick of the state this action responds to
    }

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) {
        MinecraftForge.EVENT_BUS.register(this);

        // Register the keys in the Minecraft Controls menu
        aiToggleKey = new KeyBinding("Toggle AI", Keyboard.KEY_P, "ML PvP Bot");
        ClientRegistry.registerKeyBinding(aiToggleKey);
        recordToggleKey = new KeyBinding("Toggle Recording", Keyboard.KEY_O, "ML PvP Bot");
        ClientRegistry.registerKeyBinding(recordToggleKey);

        attemptBackgroundConnection();
    }

    private void attemptBackgroundConnection() {
        if (isConnecting || out != null) return;
        isConnecting = true;

        new Thread(() -> {
            try {
                Socket socket = new Socket("127.0.0.1", aiPort);
                out = new PrintWriter(socket.getOutputStream(), true);
                in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                System.out.println("[Telemetry] Connected to Python AI Server on port " + aiPort + "!");

                // Start a nested infinite loop just for reading commands from the AI
                String line;
                while ((line = in.readLine()) != null) {
                    JsonObject json = new JsonParser().parse(line).getAsJsonObject();

                    BotAction newAction = new BotAction();
                    newAction.w = json.has("w") && json.get("w").getAsBoolean();
                    newAction.a = json.has("a") && json.get("a").getAsBoolean();
                    newAction.s = json.has("s") && json.get("s").getAsBoolean();
                    newAction.d = json.has("d") && json.get("d").getAsBoolean();
                    newAction.sprint = json.has("sprint") && json.get("sprint").getAsBoolean();

                    newAction.yaw_delta = json.has("yaw_delta") ? json.get("yaw_delta").getAsFloat() : 0f;
                    newAction.pitch_delta = json.has("pitch_delta") ? json.get("pitch_delta").getAsFloat() : 0f;

                    newAction.ack_tick = json.has("ack_tick") ? json.get("ack_tick").getAsLong() : -1;

                    newAction.jump = json.has("jump") && json.get("jump").getAsBoolean();
                    newAction.left_click = json.has("left_click") && json.get("left_click").getAsBoolean();
                    newAction.right_click = json.has("right_click") && json.get("right_click").getAsBoolean();

                    // Out-of-band reset command (teleport/heal/regear between rounds)
                    if (json.has("command")) {
                        pendingCommand = json.get("command").getAsString();
                    }

                    // Put the new command in the mailbox
                    lastActionMillis = System.currentTimeMillis();
                    currentAction = newAction;
                }
                // readLine() == null: Python closed the socket CLEANLY (normal
                // trainer shutdown/restart). Fall through to the retry below -
                // this path used to skip the catch block entirely, leaving
                // out/isConnecting set, so a clean restart of the trainer meant
                // relaunching the whole client to ever reconnect.
            } catch (Exception e) {
                // Hard drop (trainer crash, parse error) - same recovery.
            } finally {
                System.err.println("[Telemetry] Connection lost. Retrying in 5s...");
                out = null;
                in = null;
                isConnecting = false;
                try { Thread.sleep(5000); } catch (InterruptedException ignored) {}
                attemptBackgroundConnection();
            }
        }).start();
    }

    // Tab-list round-trip ping in ms; -1 until the server's player-info packet
    // arrives. OUR ping shifts WHEN our inputs register server-side (combat.py
    // extends block hold / crit timing by it); the TARGET's ping shifts how stale
    // their rendered position and reactions are - both go to Python as features.
    private int pingOf(EntityPlayer player) {
        if (mc.getNetHandler() == null) return -1;
        NetworkPlayerInfo info = mc.getNetHandler().getPlayerInfo(player.getUniqueID());
        return info != null ? info.getResponseTime() : -1;
    }

    // Distance (blocks) to the first solid block along a horizontal direction from
    // a player - "is there a wall at their back". A cornered player takes no real
    // knockback, which changes the combo/spacing math, so both fighters' back-lanes
    // (along the knockback axis attacker->victim) are reported. Scans in half-block
    // steps at shin and head height; -1 = open for at least 3 blocks.
    private double wallDistance(EntityPlayer from, double dirX, double dirZ) {
        double mag = Math.sqrt(dirX * dirX + dirZ * dirZ);
        if (mag < 1e-6) return -1.0;
        dirX /= mag; dirZ /= mag;
        for (double d = 0.5; d <= 3.0; d += 0.5) {
            double x = from.posX + dirX * d;
            double z = from.posZ + dirZ * d;
            if (isSolid(x, from.posY + 0.5, z) || isSolid(x, from.posY + 1.5, z)) {
                return d;
            }
        }
        return -1.0;
    }

    // Radial wall map: wallDistance along 8 lanes at 45 degree steps around a
    // fighter. Lane 0 points along (dirX, dirZ) - we pass the axis toward the
    // OTHER fighter, so the lanes are rotation-invariant "toward them / my back /
    // my flanks" rather than compass directions. Lane 4 therefore duplicates the
    // old single back-lane scan (my_wall/tgt_wall), which stays for checkpoint
    // compatibility. Knockback rarely lands exactly on the attacker axis (strafes,
    // aim offsets), so the diagonals are what make cornering readable.
    private void addRadialWalls(JsonObject json, String prefix, EntityPlayer p,
                                double dirX, double dirZ) {
        double mag = Math.sqrt(dirX * dirX + dirZ * dirZ);
        if (mag < 1e-6) { dirX = 1.0; dirZ = 0.0; mag = 1.0; }
        dirX /= mag; dirZ /= mag;
        for (int k = 0; k < 8; k++) {
            double ang = Math.toRadians(45.0 * k);
            double c = Math.cos(ang), s = Math.sin(ang);
            json.addProperty(prefix + k,
                    wallDistance(p, dirX * c - dirZ * s, dirX * s + dirZ * c));
        }
    }

    // Jumpable step directly ahead along (dirX, dirZ): a block at shin height
    // within a block of us, with head + above-head clear above the step, i.e. a
    // single hop puts us on top. Distinguishes "1-block step, jump it" from a
    // real wall (which reads solid at head height too and stays wallDistance's
    // business). Feeds the scripted step-up hop in combat.py - the learned jump
    // head is gone, so terrain jumps must be reflexes like the crit-jump.
    private boolean stepUpAhead(EntityPlayer p, double dirX, double dirZ) {
        double mag = Math.sqrt(dirX * dirX + dirZ * dirZ);
        if (mag < 1e-6) return false;
        dirX /= mag; dirZ /= mag;
        for (double d = 0.5; d <= 1.0; d += 0.5) {
            double x = p.posX + dirX * d;
            double z = p.posZ + dirZ * d;
            if (isSolid(x, p.posY + 0.5, z)
                    && !isSolid(x, p.posY + 1.5, z)
                    && !isSolid(x, p.posY + 2.5, z)) {
                return true;
            }
        }
        return false;
    }

    private boolean isSolid(double x, double y, double z) {
        return mc.theWorld.getBlockState(new BlockPos(x, y, z))
                .getBlock().getMaterial().blocksMovement();
    }

    // Height of a player's feet above the first solid block below them, in blocks,
    // capped at 4 (a normal jump apexes ~1.25; knockback arcs a bit higher). 0 =
    // standing on the ground. This is the "how far into a jump/knockback arc are
    // they" signal: a falling player (height shrinking) is in their crit window and
    // can't change direction; a rising one just left the ground. fallDistance would
    // miss the rising half and doesn't sync for remote players - a block scan works
    // identically for both fighters. Quarter-block steps match wallDistance's spirit.
    private double groundHeight(EntityPlayer p) {
        for (double d = 0.0; d < 4.0; d += 0.25) {
            if (isSolid(p.posX, p.posY - d - 0.05, p.posZ)) return d;
        }
        return 4.0;
    }

    // Does `looker`'s crosshair ray (out to `reach` blocks) pass through `victim`'s
    // hitbox? Ignores occluding blocks - it's "aim is on them", not "hit guaranteed".
    // For US this is exactly 1.8's hit selection (a client-side eye raytrace with
    // 3-block reach), i.e. "a click right now connects"; for the TARGET it means
    // "they can hit us the moment they click" - the trigger proactive blocking needs.
    private boolean lookRayHits(EntityPlayer looker, EntityPlayer victim, double reach) {
        Vec3 eye = new Vec3(looker.posX, looker.posY + looker.getEyeHeight(), looker.posZ);
        Vec3 look = looker.getLook(1.0F);
        Vec3 end = eye.addVector(look.xCoord * reach, look.yCoord * reach, look.zCoord * reach);
        return victim.getEntityBoundingBox().calculateIntercept(eye, end) != null;
    }

    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) {
        // END phase: the player's onUpdate() has already run this tick and copied
        // prevRotation = rotation. Restore prev to the pre-delta angle so the camera
        // interpolates the sweep across the render frames instead of snapping.
        if (event.phase == TickEvent.Phase.END) {
            if (rotationSmoothPending && mc.thePlayer != null) {
                mc.thePlayer.prevRotationYaw = preYaw;
                mc.thePlayer.prevRotationPitch = prePitch;
                rotationSmoothPending = false;
            }
            return;
        }
        if (event.phase != TickEvent.Phase.START || mc.thePlayer == null || mc.theWorld == null) return;

        // 2. The Toggle Logic
        if (recordToggleKey.isPressed()) {
            isRecording = !isRecording;
            String recMsg = isRecording ? "\u00A7e[ML Bot] RECORDING" : "\u00A77[ML Bot] RECORDING STOPPED";
            mc.thePlayer.addChatMessage(new ChatComponentText(recMsg));
        }

        if (aiToggleKey.isPressed()) {
            isAiActive = !isAiActive;
            String statusMsg = isAiActive ? "\u00A7a[ML Bot] AI ACTIVATED" : "\u00A7c[ML Bot] AI DEACTIVATED";
            mc.thePlayer.addChatMessage(new ChatComponentText(statusMsg));

            // Release all keys if we turn the AI off mid-fight to prevent ghost walking
            if (!isAiActive) {
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), false);
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), false);
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), false);
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), false);
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), false);
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindJump.getKeyCode(), false);
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
                rotationSmoothPending = false;  // hand the camera back to vanilla cleanly
            }
        }

        if (out == null) return;

        // Telemetry must flow EVERY tick, GUI or not - Python blocks on recv, so a
        // menu pausing telemetry on this client stalls the whole trainer. Note this
        // stream is NOT lockstep: states go out every tick whether or not Python has
        // replied, so Python drains to the freshest frame (see pvp_env._recv_state)
        // instead of consuming a stale backlog. A GUI only suppresses APPLYING
        // inputs (guiOpen below), never sending.
        boolean deathScreen = mc.currentScreen instanceof GuiGameOver;
        boolean guiOpen = mc.currentScreen != null && !deathScreen;
        // Nobody is around to click Respawn during self-play; click it ourselves.
        if (deathScreen && isAiActive && mc.thePlayer.getHealth() <= 0.0F) {
            mc.thePlayer.respawnPlayer();
        }

        // Run any queued reset commands on the client thread (Minecraft isn't
        // thread-safe). Python joins them with ";" - split back into one chat
        // message each, or the server parses the whole line as a single command.
        if (pendingCommand != null) {
            String cmd = pendingCommand;
            pendingCommand = null;
            for (String part : cmd.split(";")) {
                String c = part.trim();
                if (!c.isEmpty()) mc.thePlayer.sendChatMessage(c);
            }
        }

        EntityPlayer target = null;
        double closestDist = Double.MAX_VALUE;

        // 3. Scan for closest Human Player (to send telemetry to AI)
        for (EntityPlayer player : mc.theWorld.playerEntities) {
            if (player != mc.thePlayer && !player.isInvisible()) {
                double dist = mc.thePlayer.getDistanceToEntity(player);
                if (dist < closestDist) {
                    closestDist = dist;
                    target = player;
                }
            }
        }

        // 4. Send the State to Python
        final long thisTick = tickCounter++;
        try {
            JsonObject json = new JsonObject();

            // Session bookkeeping - lets Python detect gaps when stacking frames
            json.addProperty("tick", thisTick);
            json.addProperty("recording", isRecording);
            // Whether the bot is driving this client (P toggles it). When false a HUMAN
            // took over, so the self-play trainer must not attribute this round to the
            // scripted style it scheduled (the actions it sends are being ignored).
            json.addProperty("ai_active", isAiActive);
            // Last measured control-loop lag (ticks); healthy 1, phase-locked 2,
            // -1 until Python starts echoing ack_tick. Lets the trainer log it.
            json.addProperty("staleness", lastStaleness);

            // Game State
            json.addProperty("player_x", mc.thePlayer.posX);
            json.addProperty("player_y", mc.thePlayer.posY);
            json.addProperty("player_z", mc.thePlayer.posZ);
            json.addProperty("player_yaw", mc.thePlayer.rotationYaw);
            json.addProperty("player_pitch", mc.thePlayer.rotationPitch);
            json.addProperty("on_ground", mc.thePlayer.onGround);
            json.addProperty("my_vx", mc.thePlayer.posX - mc.thePlayer.prevPosX);
            json.addProperty("my_vz", mc.thePlayer.posZ - mc.thePlayer.prevPosZ);
            json.addProperty("my_vy", mc.thePlayer.posY - mc.thePlayer.prevPosY);
            json.addProperty("my_hurt", mc.thePlayer.hurtTime);
            json.addProperty("my_health", mc.thePlayer.getHealth());
            // Sprint = the knockback-bonus flag; block = using item (half damage,
            // 20% movement). Both are entity state, so they read correctly for the
            // remote player too (use-item syncs via the eating flag, sprint via
            // entity flags - that's how you can SEE other players block/sprint).
            json.addProperty("my_sprint", mc.thePlayer.isSprinting());
            json.addProperty("my_block", mc.thePlayer.isBlocking());
            json.addProperty("my_ping", pingOf(mc.thePlayer));
            json.addProperty("my_ground_h", groundHeight(mc.thePlayer));

            if (target != null) {
                json.addProperty("target_x", target.posX);
                json.addProperty("target_y", target.posY);
                json.addProperty("target_z", target.posZ);
                json.addProperty("target_dist", closestDist);
                json.addProperty("tgt_vx", target.posX - target.prevPosX);
                json.addProperty("tgt_vy", target.posY - target.prevPosY);
                json.addProperty("tgt_vz", target.posZ - target.prevPosZ);
                json.addProperty("tgt_hurt", target.hurtTime);
                json.addProperty("tgt_health", target.getHealth());
                json.addProperty("tgt_sprint", target.isSprinting());
                json.addProperty("tgt_block", target.isBlocking());
                json.addProperty("tgt_ping", pingOf(target));
                // Swing animation syncs to all clients - this is "they're attacking
                // RIGHT NOW", the signal proactive blocking has to key off.
                json.addProperty("tgt_swing", target.isSwingInProgress);
                // Where the target is looking (features.py turns this into
                // "how far are they looking away from me")
                json.addProperty("tgt_yaw", target.rotationYaw);
                // Back-lanes along the knockback axis: a hit launches the victim
                // AWAY from the attacker, so scan behind each fighter on that line.
                double kbX = target.posX - mc.thePlayer.posX;
                double kbZ = target.posZ - mc.thePlayer.posZ;
                json.addProperty("tgt_wall", wallDistance(target, kbX, kbZ));
                json.addProperty("my_wall", wallDistance(mc.thePlayer, -kbX, -kbZ));
                // Full radial wall map (v7): lane 0 = toward the other fighter,
                // 45-degree steps; lane 4 = the back/knockback lane above.
                addRadialWalls(json, "my_rwall_", mc.thePlayer, kbX, kbZ);
                addRadialWalls(json, "tgt_rwall_", target, -kbX, -kbZ);
                // 1-block jumpable step between us and them (scripted hop trigger)
                json.addProperty("my_step_up", stepUpAhead(mc.thePlayer, kbX, kbZ));
                // Remote onGround is driven by the server's movement packets - it's
                // the airborne/juggle state (air friction 0.91 vs ~0.546 grounded,
                // so airborne victims fly ~4x further per hit).
                json.addProperty("tgt_on_ground", target.onGround);
                json.addProperty("my_ray_hit", lookRayHits(mc.thePlayer, target, 3.0));
                json.addProperty("tgt_ray_hit", lookRayHits(target, mc.thePlayer, 3.0));
                json.addProperty("tgt_ground_h", groundHeight(target));
            } else {
                json.addProperty("target_dist", -1.0);
            }

            // Human inputs - what the logger records as training labels
            json.addProperty("in_w", mc.gameSettings.keyBindForward.isKeyDown());
            json.addProperty("in_a", mc.gameSettings.keyBindLeft.isKeyDown());
            json.addProperty("in_s", mc.gameSettings.keyBindBack.isKeyDown());
            json.addProperty("in_d", mc.gameSettings.keyBindRight.isKeyDown());
            // actual sprint state, not the keybind - double-tap-W sprinting never
            // presses keyBindSprint, so reading the key logs sprint as always-off
            json.addProperty("in_sprint", mc.thePlayer.isSprinting());
            json.addProperty("in_jump", mc.gameSettings.keyBindJump.isKeyDown());
            json.addProperty("in_left_click", mc.gameSettings.keyBindAttack.isKeyDown());
            json.addProperty("in_right_click", mc.gameSettings.keyBindUseItem.isKeyDown());

            out.println(json.toString());
        } catch (Exception e) {
            out = null;
        }

        // --- APPLY AI ACTIONS ---
        // Skipped while a menu is open (telemetry above still ran, keeping lockstep);
        // release the movement keys so the bot doesn't ghost-walk under the GUI.
        if (isAiActive && guiOpen) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindJump.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
        }
        if (isAiActive && !guiOpen) {
            // Anti-phase-lock: the action we SHOULD apply this tick is the reply to
            // last tick's state (ack == thisTick-1). If it hasn't arrived yet - the
            // reply is in flight because the other client's tick clock sits just
            // before ours - wait briefly for it rather than applying a stale action.
            // lag > 4 means Python is paused (PPO update / reset), not phase-lagged;
            // ack < 0 means Python isn't echoing (old server) - don't wait for those.
            // Watchdog first (see STALE_ACTION_MS): if Python has gone quiet, don't
            // wait on it and don't replay the held keys - act idle until it's back.
            boolean pythonPaused = lastActionMillis > 0
                    && System.currentTimeMillis() - lastActionMillis > STALE_ACTION_MS;
            long lag = (thisTick - 1) - currentAction.ack_tick;
            if (!pythonPaused && currentAction.ack_tick >= 0 && lag >= 1 && lag <= 4) {
                int waited = 0;
                while (waited < FRESH_WAIT_MS
                        && (thisTick - 1) - currentAction.ack_tick >= 1) {
                    try { Thread.sleep(1); } catch (InterruptedException e) { break; }
                    waited++;
                }
                staleWaitedMs += waited;
            }
            // A fresh BotAction is all-false/zero (ack -1, so staleness bookkeeping
            // below skips it) - exactly "release everything".
            BotAction action = pythonPaused ? new BotAction() : currentAction;

            // Staleness bookkeeping: summary line every 200 ticks (~10 s).
            if (action.ack_tick >= 0) {
                lastStaleness = thisTick - action.ack_tick;
                staleSum += lastStaleness;
                staleMax = Math.max(staleMax, lastStaleness);
                staleTicks++;
                if (staleTicks >= 200) {
                    System.out.println(String.format(
                        "[mlpvp:%d] staleness avg %.2f max %d over %d ticks (waited %d ms)",
                        aiPort, staleSum / (double) staleTicks, staleMax, staleTicks,
                        staleWaitedMs));
                    staleSum = staleMax = staleTicks = staleWaitedMs = 0;
                }
            }

            KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), action.w);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), action.a);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), action.s);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), action.d);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindJump.getKeyCode(), action.jump);

            // Sprint: drive the entity flag directly, not the keybind. A toggle-sprint
            // setup never holds keyBindSprint, so writing that key's state does nothing
            // (or fights the toggle). setSprinting is what actually gates the 1.3x speed
            // and the sprint knockback we W-tap to reset.
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), action.sprint);
            mc.thePlayer.setSprinting(action.sprint && action.w);

            // Remember the pre-delta angle so onClientTick END can restore prevRotation
            // and the camera interpolates this tick's rotation smoothly (see field docs).
            preYaw = mc.thePlayer.rotationYaw;
            prePitch = mc.thePlayer.rotationPitch;
            mc.thePlayer.rotationYaw += (action.yaw_delta * aiAimSensitivity);
            mc.thePlayer.rotationPitch += (action.pitch_delta * aiAimSensitivity);
            // Vanilla mouse input can never push pitch past vertical; neither may the AI
            mc.thePlayer.rotationPitch = Math.max(-90f, Math.min(90f, mc.thePlayer.rotationPitch));
            rotationSmoothPending = true;

            // Attack is an edge - onTick queues exactly one clickMouse() next runTick.
            if (action.left_click) {
                KeyBinding.onTick(mc.gameSettings.keyBindAttack.getKeyCode());
            }
            // Blocking is a HOLD. onTick fires a single press, which is why sword-block
            // never engaged; vanilla only blocks while keyBindUseItem.isKeyDown().
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), action.right_click);

            // Consume impulses so we don't get the helicopter neck bug
            action.yaw_delta = 0;
            action.pitch_delta = 0;
            action.left_click = false;
            action.jump = false;
        }
    }
}