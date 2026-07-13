package net.soupchugger.ml;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.ChatComponentText;
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

    // 1. Create the Keybinding and State Tracker
    private KeyBinding recordKey;
    private boolean isAiActive = false;

    // --- NEW: AI Aim Sensitivity Multiplier ---
    // Boosts the AI's smoothed mouse movements to combat MSE regression lag
    private float aiAimSensitivity = 2.5f;

    // The "Mailbox" - Volatile ensures thread safety when reading/writing
    private volatile BotAction currentAction = new BotAction();

    // Data container for our inputs
    public static class BotAction {
        public boolean w, a, s, d, sprint;
        public float yaw_delta, pitch_delta;
        public boolean left_click, right_click;
    }

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) {
        MinecraftForge.EVENT_BUS.register(this);

        // Register the 'P' key in the Minecraft Controls menu
        recordKey = new KeyBinding("Toggle AI", Keyboard.KEY_P, "ML PvP Bot");
        ClientRegistry.registerKeyBinding(recordKey);

        attemptBackgroundConnection();
    }

    private void attemptBackgroundConnection() {
        if (isConnecting || out != null) return;
        isConnecting = true;

        new Thread(() -> {
            try {
                Socket socket = new Socket("127.0.0.1", 9999);
                out = new PrintWriter(socket.getOutputStream(), true);
                in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                System.out.println("[Telemetry] Connected to Python AI Server!");

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

                    newAction.left_click = json.has("left_click") && json.get("left_click").getAsBoolean();
                    newAction.right_click = json.has("right_click") && json.get("right_click").getAsBoolean();

                    // Put the new command in the mailbox
                    currentAction = newAction;
                }
            } catch (Exception e) {
                System.err.println("[Telemetry] Connection lost. Retrying...");
                out = null;
                in = null;
                isConnecting = false;
                try { Thread.sleep(5000); } catch (InterruptedException ignored) {}
                attemptBackgroundConnection();
            }
        }).start();
    }

    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) {
        if (event.phase != TickEvent.Phase.START || mc.thePlayer == null || mc.theWorld == null) return;

        // 2. The Toggle Logic
        if (recordKey.isPressed()) {
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
            }
        }

        // If a GUI is open or socket is dead, freeze
        if (mc.currentScreen != null || out == null) return;

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
        try {
            JsonObject json = new JsonObject();

            // Game State
            json.addProperty("player_x", mc.thePlayer.posX);
            json.addProperty("player_y", mc.thePlayer.posY);
            json.addProperty("player_z", mc.thePlayer.posZ);
            json.addProperty("player_yaw", mc.thePlayer.rotationYaw);
            json.addProperty("player_pitch", mc.thePlayer.rotationPitch);
            json.addProperty("on_ground", mc.thePlayer.onGround);

            if (target != null) {
                json.addProperty("target_x", target.posX);
                json.addProperty("target_y", target.posY);
                json.addProperty("target_z", target.posZ);
                json.addProperty("target_dist", closestDist);
            } else {
                json.addProperty("target_dist", -1.0);
            }

            out.println(json.toString());
        } catch (Exception e) {
            out = null;
        }

        // --- APPLY AI ACTIONS ---
        if (isAiActive) {
            BotAction action = currentAction;

            KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), action.w);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), action.a);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), action.s);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), action.d);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), action.sprint);

            // Apply the sensitivity multiplier to counteract PyTorch MSE smoothing
            mc.thePlayer.rotationYaw += (action.yaw_delta * aiAimSensitivity);
            mc.thePlayer.rotationPitch += (action.pitch_delta * aiAimSensitivity);

            if (action.left_click) {
                KeyBinding.onTick(mc.gameSettings.keyBindAttack.getKeyCode());
            }
            if (action.right_click) {
                KeyBinding.onTick(mc.gameSettings.keyBindUseItem.getKeyCode());
            }

            // Consume impulses so we don't get the helicopter neck bug
            action.yaw_delta = 0;
            action.pitch_delta = 0;
            action.left_click = false;
            action.right_click = false;
        }
    }
}