using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

/// <summary>
/// Bridges Unity camera frames and vehicle telemetry to the
/// Python perception server over UDP.
/// Sends: JPEG frame + telemetry JSON on port 5006
/// Receives: detection + control JSON on port 5008
/// Applies autonomous control signals to RCC vehicle.
/// </summary>
public class PerceptionBridge : MonoBehaviour
{
    [Header("Network Settings")]
    public string pythonIP   = "127.0.0.1";
    public int sendPort      = 5006;
    public int receivePort   = 5008;

    [Header("Camera")]
    public Camera perceptionCamera;
    public int captureWidth  = 640;
    public int captureHeight = 360;
    public int jpegQuality   = 80;

    [Header("Capture Settings")]
    public int captureEveryNFrames = 6;
    public bool sendFrames = true;

    [Header("Vehicle Reference")]
    public GameObject playerVehicle;

    [Header("Autonomous Control")]
    public bool autonomousEnabled  = false;
    public float steeringSmoothing = 0.3f;
    public float throttleSmoothing = 0.5f;

    // ── NETWORK ──────────────────────────────────────────
    private UdpClient sendClient;
    private UdpClient receiveClient;
    private IPEndPoint pythonEndPoint;

    // ── RENDER TEXTURE ───────────────────────────────────
    private RenderTexture captureRT;
    private Texture2D captureTexture;

    // ── STATE ────────────────────────────────────────────
    private int frameCount = 0;
    private bool isRunning = false;

    // ── VEHICLE TELEMETRY ─────────────────────────────────
    private float vehicleSpeed     = 0f;
    private float vehicleSteering  = 0f;
    private int   vehicleGear      = 0;
    private Vector3 vehiclePosition = Vector3.zero;

    // ── AUTONOMOUS CONTROL (from Python) ─────────────────
    private float targetSteering = 0f;
    private float targetThrottle = 0f;
    private float targetBrake    = 0f;
    private bool  emergencyBrake = false;

    // Smoothed values applied to vehicle
    private float smoothedSteering = 0f;
    private float smoothedThrottle = 0f;
    private float smoothedBrake    = 0f;

    // ── LATEST RESULT ────────────────────────────────────
    private PerceptionResult latestResult;
    private string latestJson = "";

    // ── DATA STRUCTURES ──────────────────────────────────

    [Serializable]
    public class BoundingBox
    {
        public float x1, y1, x2, y2;
        public float cx, cy, w, h;
    }

    [Serializable]
    public class Detection
    {
        public int    class_id;
        public string class_name;
        public float  confidence;
        public BoundingBox bbox;
    }

    [Serializable]
    public class SceneState
    {
        public int   vehicle_count;
        public int   pedestrian_count;
        public int   total_tracked;
        public float closest_vehicle_distance;
        public float highest_risk_score;
        public bool  emergency_brake;
        public bool  collision_warning;
        public float ego_speed;
        public float ego_steering;
    }

    [Serializable]
    public class ControlSignals
    {
        public float steering;
        public float throttle;
        public float brake;
        public int   mode;
        public bool  emergency_brake;
    }

    [Serializable]
    public class PerceptionResult
    {
        public int        frame_id;
        public float      inference_ms;
        public int        detection_count;
        public Detection[]  detections;
        public float      lane_offset;
        public bool       lane_detected;
        public SceneState scene_state;
        public ControlSignals control;
    }

    // ── UNITY LIFECYCLE ──────────────────────────────────

    void Start()
    {
        InitialiseNetwork();
        InitialiseRenderTexture();
        isRunning = true;
        Debug.Log("[PerceptionBridge] Started → " +
                  pythonIP + ":" + sendPort);
    }

    void Update()
    {
        UpdateTelemetry();

        // Send frame every N frames
        frameCount++;
        if (sendFrames && frameCount % captureEveryNFrames == 0)
            StartCoroutine(CaptureAndSend());

        // Non-blocking receive check every frame
        TryReceive();

        // Apply autonomous control if enabled
        if (autonomousEnabled)
            ApplyAutonomousControl();

        // Debug log every 60 frames
        if (frameCount % 60 == 0)
        {
            Debug.Log(
                $"[PerceptionBridge] " +
                $"HasResult:{latestResult != null} | " +
                $"Throttle:{targetThrottle:F2} | " +
                $"Steer:{targetSteering:F2} | " +
                $"Brake:{targetBrake:F2}");
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        sendClient?.Close();
        receiveClient?.Close();
        if (captureRT != null) captureRT.Release();
        if (captureTexture != null) Destroy(captureTexture);
        Debug.Log("[PerceptionBridge] Stopped.");
    }

    // ── INITIALISATION ───────────────────────────────────

    void InitialiseNetwork()
    {
        // Send client — sends frames to Python
        sendClient    = new UdpClient();
        pythonEndPoint = new IPEndPoint(
            IPAddress.Parse(pythonIP), sendPort);

        // Receive client — non-blocking, polls in Update()
        receiveClient = new UdpClient(receivePort);
        receiveClient.Client.Blocking = false;

        Debug.Log("[PerceptionBridge] Network ready. " +
                  "Receiving on port " + receivePort);
    }

    void InitialiseRenderTexture()
    {
        captureRT = new RenderTexture(
            captureWidth, captureHeight, 24,
            RenderTextureFormat.ARGB32);

        captureTexture = new Texture2D(
            captureWidth, captureHeight,
            TextureFormat.RGB24, false);

        if (perceptionCamera != null)
            perceptionCamera.targetTexture = captureRT;
        else
            Debug.LogWarning(
                "[PerceptionBridge] No camera assigned!");
    }

    // ── TELEMETRY ────────────────────────────────────────

    void UpdateTelemetry()
    {
        if (playerVehicle == null) return;

        var rcc = playerVehicle
            .GetComponent<RCC_CarControllerV3>();
        if (rcc != null)
        {
            vehicleSpeed    = rcc.speed;
            vehicleSteering = rcc.steerInput;
            vehicleGear     = rcc.currentGear;
        }
        else
        {
            var rb = playerVehicle.GetComponent<Rigidbody>();
            if (rb != null)
                vehicleSpeed = rb.linearVelocity.magnitude;
        }
        vehiclePosition = playerVehicle.transform.position;
    }

    // ── NON-BLOCKING RECEIVE ─────────────────────────────

    void TryReceive()
    {
        
        try
        {
            // Available > 0 means data is waiting
            // Non-blocking so no waiting if nothing there
            while (receiveClient.Available > 0)
            {
                IPEndPoint remote = new IPEndPoint(
                    IPAddress.Any, 0);
                byte[] data = receiveClient.Receive(
                    ref remote);
                string json = Encoding.UTF8.GetString(data);

                Debug.Log(
                    $"[PerceptionBridge] Received " +
                    $"{data.Length} bytes from Python");

                PerceptionResult result =
                    JsonUtility.FromJson<PerceptionResult>(
                        json);

                latestResult = result;
                latestJson   = json;

                Debug.Log("[PerceptionBridge] JSON preview: " + 
          json.Substring(0, Mathf.Min(300, json.Length)));

                // Apply control signals if autonomous enabled
                if (result != null &&
                    result.control != null &&
                    autonomousEnabled)
                {
                    targetSteering = result.control.steering;
                    targetThrottle = result.control.throttle;
                    targetBrake    = result.control.brake;
                    emergencyBrake =
                        result.control.emergency_brake;

                    Debug.Log(
                        $"[PerceptionBridge] Control → " +
                        $"Steer:{targetSteering:F3} | " +
                        $"Thr:{targetThrottle:F2} | " +
                        $"Brk:{targetBrake:F2}");
                }
            }
        }
        catch (SocketException se)
        {
            // WouldBlock = no data available, normal
            if (se.SocketErrorCode !=
                SocketError.WouldBlock)
            {
                Debug.LogWarning(
                    "[PerceptionBridge] Socket: " +
                    se.Message);
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning(
                "[PerceptionBridge] TryReceive: " +
                e.Message);
        }
    }

    // ── AUTONOMOUS CONTROL APPLICATION ───────────────────

    void ApplyAutonomousControl()
    {
        if (playerVehicle == null) return;

        var rcc = playerVehicle
            .GetComponent<RCC_CarControllerV3>();
        if (rcc == null) return;

        // Ensure engine is running
        if (!rcc.engineRunning)
            rcc.StartEngine();

        // Smooth control signals
        smoothedSteering = Mathf.Lerp(
            smoothedSteering, targetSteering,
            steeringSmoothing);
        smoothedThrottle = Mathf.Lerp(
            smoothedThrottle, targetThrottle,
            throttleSmoothing);
        smoothedBrake = Mathf.Lerp(
            smoothedBrake, targetBrake,
            throttleSmoothing);

        // Apply to RCC
        rcc.externalController = true;
        rcc.gasInput           = smoothedThrottle;
        rcc.brakeInput         = smoothedBrake;
        rcc.steerInput         = smoothedSteering;
        rcc.handbrakeInput     = 0f;

        // Debug every 60 frames
        if (frameCount % 60 == 0)
        {
            Debug.Log(
                $"[PerceptionBridge] RCC Applied → " +
                $"Gas:{rcc.gasInput:F2} | " +
                $"Brake:{rcc.brakeInput:F2} | " +
                $"Steer:{rcc.steerInput:F2} | " +
                $"Engine:{rcc.engineRunning} | " +
                $"Speed:{rcc.speed:F1}");
        }
    }

    // ── CAPTURE AND SEND ─────────────────────────────────

    IEnumerator CaptureAndSend()
    {
        yield return new WaitForEndOfFrame();

        if (perceptionCamera == null || captureRT == null)
            yield break;

        RenderTexture.active = captureRT;
        captureTexture.ReadPixels(
            new Rect(0, 0, captureWidth, captureHeight),
            0, 0);
        captureTexture.Apply();
        RenderTexture.active = null;

        byte[] jpegBytes     = captureTexture.EncodeToJPG(
            jpegQuality);
        string telemetryJson = BuildTelemetryJson();
        byte[] telemetryBytes = Encoding.UTF8.GetBytes(
            telemetryJson);

        int    len    = telemetryBytes.Length;
        byte[] packet = new byte[
            4 + len + jpegBytes.Length];

        // Big-endian uint32 header
        packet[0] = (byte)((len >> 24) & 0xFF);
        packet[1] = (byte)((len >> 16) & 0xFF);
        packet[2] = (byte)((len >> 8)  & 0xFF);
        packet[3] = (byte)((len)       & 0xFF);

        Buffer.BlockCopy(
            telemetryBytes, 0, packet, 4, len);
        Buffer.BlockCopy(
            jpegBytes, 0, packet,
            4 + len, jpegBytes.Length);

        try
        {
            sendClient.Send(
                packet, packet.Length, pythonEndPoint);
        }
        catch (Exception e)
        {
            Debug.LogWarning(
                "[PerceptionBridge] Send: " + e.Message);
        }
    }

    string BuildTelemetryJson()
    {
        float px = vehiclePosition.x;
        float py = vehiclePosition.y;
        float pz = vehiclePosition.z;

        return "{" +
            "\"speed\":"    + vehicleSpeed.ToString("F2")    + "," +
            "\"steering\":" + vehicleSteering.ToString("F3") + "," +
            "\"gear\":"     + vehicleGear                    + "," +
            "\"pos_x\":"    + px.ToString("F2")              + "," +
            "\"pos_y\":"    + py.ToString("F2")              + "," +
            "\"pos_z\":"    + pz.ToString("F2")              +
            "}";
    }

    // ── PUBLIC API ───────────────────────────────────────

    public void EnableAutonomous(bool enable)
    {
        autonomousEnabled = enable;

        if (!enable)
        {
            var rcc = playerVehicle?
                .GetComponent<RCC_CarControllerV3>();
            if (rcc != null)
            {
                rcc.externalController = false;
                rcc.gasInput       = 0f;
                rcc.brakeInput     = 0f;
                rcc.steerInput     = 0f;
                rcc.handbrakeInput = 0f;
            }
            smoothedSteering = 0f;
            smoothedThrottle = 0f;
            smoothedBrake    = 0f;
        }

        Debug.Log("[PerceptionBridge] Autonomous: " + enable);
    }

    public PerceptionResult GetLatestResult() =>
        latestResult;

    public bool HasResult() => latestResult != null;

    public SceneState GetSceneState() =>
        latestResult?.scene_state;

    // ── DEBUG HUD ────────────────────────────────────────

    void OnGUI()
    {
        if (!Application.isPlaying) return;

        GUIStyle style    = new GUIStyle(GUI.skin.label);
        style.fontSize    = 14;
        style.richText    = true;
        style.normal.textColor = Color.white;

        string modeStr = autonomousEnabled
            ? "<color=lime>AUTONOMOUS</color>"
            : "<color=yellow>MANUAL</color>";

        GUI.Label(new Rect(10, 10, 400, 25),
            $"Mode: {modeStr}", style);

        PerceptionResult r = latestResult;
        if (r == null)
        {
            GUI.color = Color.red;
            GUI.Label(new Rect(10, 35, 400, 20),
                "Waiting for Python server...");
            return;
        }

        GUI.color = Color.green;
        GUI.Label(new Rect(10, 35, 400, 20),
            $"Detections: {r.detection_count} | " +
            $"Inference: {r.inference_ms:F1}ms");

        GUI.Label(new Rect(10, 55, 400, 20),
            $"Lane: {r.lane_offset:F3} | " +
            $"{(r.lane_detected ? "DETECTED" : "NONE")}");

        if (r.scene_state != null)
        {
            GUI.color = r.scene_state.collision_warning
                ? Color.red : Color.green;
            GUI.Label(new Rect(10, 75, 400, 20),
                $"Closest: " +
                $"{r.scene_state.closest_vehicle_distance:F1}m" +
                $" | Tracked: {r.scene_state.total_tracked}");

            if (r.scene_state.emergency_brake)
            {
                GUI.color = Color.red;
                GUI.Label(new Rect(10, 95, 300, 20),
                    "EMERGENCY BRAKE");
            }
        }

        if (autonomousEnabled && r.control != null)
        {
            GUI.color = Color.cyan;
            GUI.Label(new Rect(10, 115, 400, 20),
                $"Steer:{r.control.steering:+0.000} | " +
                $"Thr:{r.control.throttle:0.00} | " +
                $"Brk:{r.control.brake:0.00}");
        }

        if (r.detections == null) return;
        GUI.color = Color.yellow;
        int y = 140;
        foreach (var det in r.detections)
        {
            GUI.Label(new Rect(10, y, 300, 18),
                $"{det.class_name} {det.confidence:F2}");
            y += 18;
        }
    }
}