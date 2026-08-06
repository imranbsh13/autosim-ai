using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Bridges Unity camera frames and vehicle telemetry to the
/// Python perception server over UDP.
/// Sends: JPEG frame + telemetry JSON on port 5006
/// Receives: detection JSON on port 5007
/// </summary>
public class PerceptionBridge : MonoBehaviour
{
    [Header("Network Settings")]
    public string pythonIP = "127.0.0.1";
    public int sendPort = 5006;
    public int receivePort = 5007;

    [Header("Camera")]
    public Camera perceptionCamera;
    public int captureWidth = 640;
    public int captureHeight = 360;
    public int jpegQuality = 80;

    [Header("Capture Settings")]
    public int captureEveryNFrames = 6;
    public bool sendFrames = true;

    [Header("Vehicle Reference")]
    public GameObject playerVehicle;

    // Network
    private UdpClient sendClient;
    private UdpClient receiveClient;
    private Thread receiveThread;
    private IPEndPoint pythonEndPoint;

    // Render texture for camera capture
    private RenderTexture captureRT;
    private Texture2D captureTexture;

    // State
    private int frameCount = 0;
    private bool isRunning = false;

    // Latest detection result from Python
    private string latestDetectionJson = "";
    private DetectionResult latestResult;
    private readonly object resultLock = new object();

    // Telemetry — populated from RCC
    private float vehicleSpeed = 0f;
    private float vehicleSteering = 0f;
    private int vehicleGear = 0;
    private Vector3 vehiclePosition = Vector3.zero;

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
        public int class_id;
        public string class_name;
        public float confidence;
        public BoundingBox bbox;
    }

    [Serializable]
    public class DetectionResult
    {
        public int frame_id;
        public float inference_ms;
        public int detection_count;
        public Detection[] detections;
    }

    // ── UNITY LIFECYCLE ──────────────────────────────────

    void Start()
    {
        InitialiseNetwork();
        InitialiseRenderTexture();
        isRunning = true;
        Debug.Log("[PerceptionBridge] Started. Sending to " +
                  pythonIP + ":" + sendPort);
    }

    void Update()
    {
        // Update telemetry from vehicle
        UpdateTelemetry();

        // Send frame every N frames
        frameCount++;
        if (sendFrames && frameCount % captureEveryNFrames == 0)
        {
            StartCoroutine(CaptureAndSend());
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        receiveThread?.Abort();
        sendClient?.Close();
        receiveClient?.Close();

        if (captureRT != null) captureRT.Release();
        if (captureTexture != null) Destroy(captureTexture);

        Debug.Log("[PerceptionBridge] Stopped.");
    }

    // ── INITIALISATION ───────────────────────────────────

    void InitialiseNetwork()
    {
        // Send client
        sendClient = new UdpClient();
        pythonEndPoint = new IPEndPoint(
            IPAddress.Parse(pythonIP), sendPort);

        // Receive client
        receiveClient = new UdpClient(receivePort);
        receiveClient.Client.ReceiveTimeout = 100;

        // Start receive thread
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();

        Debug.Log("[PerceptionBridge] Network initialised. " +
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
        {
            perceptionCamera.targetTexture = captureRT;
        }
        else
        {
            Debug.LogWarning("[PerceptionBridge] No camera assigned!");
        }
    }

    // ── TELEMETRY ────────────────────────────────────────

    void UpdateTelemetry()
    {
        if (playerVehicle == null) return;

        // Try to get RCC component for accurate vehicle data
        var rcc = playerVehicle.GetComponent
            <RCC_CarControllerV3>();

        if (rcc != null)
        {
            vehicleSpeed    = rcc.speed;
            vehicleSteering = rcc.steerInput;
            vehicleGear     = rcc.currentGear;
        }
        else
        {
            // Fallback: estimate speed from Rigidbody
            var rb = playerVehicle.GetComponent<Rigidbody>();
            if (rb != null)
                vehicleSpeed = rb.linearVelocity.magnitude;
        }

        vehiclePosition = playerVehicle.transform.position;
    }

    // ── CAPTURE AND SEND ─────────────────────────────────

    IEnumerator CaptureAndSend()
    {
        // Wait for end of frame so rendering is complete
        yield return new WaitForEndOfFrame();

        if (perceptionCamera == null || captureRT == null) yield break;

        // Read pixels from render texture
        RenderTexture.active = captureRT;
        captureTexture.ReadPixels(
            new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        captureTexture.Apply();
        RenderTexture.active = null;

        // Encode to JPEG
        byte[] jpegBytes = captureTexture.EncodeToJPG(jpegQuality);

        // Build telemetry JSON
        string telemetryJson = BuildTelemetryJson();
        byte[] telemetryBytes = Encoding.UTF8.GetBytes(telemetryJson);

        // Build packet:
        // [4 bytes: telemetry length] [telemetry JSON] [JPEG bytes]
        int telemetryLen = telemetryBytes.Length;
        byte[] packet = new byte[4 + telemetryLen + jpegBytes.Length];

        // Write telemetry length as big-endian uint32
        packet[0] = (byte)((telemetryLen >> 24) & 0xFF);
        packet[1] = (byte)((telemetryLen >> 16) & 0xFF);
        packet[2] = (byte)((telemetryLen >> 8) & 0xFF);
        packet[3] = (byte)((telemetryLen) & 0xFF);

        // Write telemetry JSON
        Buffer.BlockCopy(telemetryBytes, 0, packet, 4, telemetryLen);

        // Write JPEG
        Buffer.BlockCopy(jpegBytes, 0, packet,
            4 + telemetryLen, jpegBytes.Length);

        // Send over UDP
        try
        {
            sendClient.Send(packet, packet.Length, pythonEndPoint);
            Debug.Log($"[PerceptionBridge] Sent frame {frameCount}, " +
          $"packet size: {packet.Length} bytes");
        }
        catch (Exception e)
        {
            Debug.LogWarning("[PerceptionBridge] Send failed: " + e.Message);
        }
    }

    string BuildTelemetryJson()
    {
        return $"{{" +
               $"\"speed\":{vehicleSpeed:F2}," +
               $"\"steering\":{vehicleSteering:F3}," +
               $"\"gear\":{vehicleGear}," +
               $"\"pos_x\":{vehiclePosition.x:F2}," +
               $"\"pos_y\":{vehiclePosition.y:F2}," +
               $"\"pos_z\":{vehiclePosition.z:F2}" +
               $"}}";
    }

    // ── RECEIVE LOOP ─────────────────────────────────────

    void ReceiveLoop()
    {
        IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, 0);

        while (isRunning)
        {
            try
            {
                byte[] data = receiveClient.Receive(ref remoteEP);
                string json = Encoding.UTF8.GetString(data);

                // Parse detection result
                DetectionResult result =
                    JsonUtility.FromJson<DetectionResult>(json);

                lock (resultLock)
                {
                    latestResult = result;
                    latestDetectionJson = json;
                }
            }
            catch (SocketException)
            {
                // Timeout — normal, just keep looping
            }
            catch (Exception e)
            {
                if (isRunning)
                    Debug.LogWarning(
                        "[PerceptionBridge] Receive error: " + e.Message);
            }
        }
    }

    // ── PUBLIC ACCESSORS ─────────────────────────────────

    /// <summary>
    /// Get the latest detection result from Python.
    /// Call from other scripts to read detections.
    /// </summary>
    public DetectionResult GetLatestResult()
    {
        lock (resultLock)
        {
            return latestResult;
        }
    }

    /// <summary>
    /// Get the raw JSON string of the latest result.
    /// Useful for debugging.
    /// </summary>
    public string GetLatestJson()
    {
        lock (resultLock)
        {
            return latestDetectionJson;
        }
    }

    /// <summary>
    /// Returns true if we have received at least one
    /// detection result from Python.
    /// </summary>
    public bool HasResult()
    {
        lock (resultLock)
        {
            return latestResult != null;
        }
    }

    // ── DEBUG GIZMOS ─────────────────────────────────────

    void OnGUI()
    {
        if (!Application.isPlaying) return;

        DetectionResult result = GetLatestResult();
        if (result == null) return;

        // Show detection count in top-left corner
        GUI.color = Color.green;
        GUI.Label(new Rect(10, 10, 300, 25),
            $"Detections: {result.detection_count} | " +
            $"Inference: {result.inference_ms:F1}ms");

        if (result.detections == null) return;

        // Draw detection labels
        GUI.color = Color.yellow;
        int y = 35;
        foreach (var det in result.detections)
        {
            GUI.Label(new Rect(10, y, 300, 20),
                $"{det.class_name} {det.confidence:F2}");
            y += 20;
        }
    }
}