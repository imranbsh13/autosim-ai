using System.Collections;
using System.Collections.Generic;
using UnityEngine;


/// <summary>
/// Runs YOLOv8 inference on-device using Unity Sentis.
/// Processes camera frames and outputs bounding box detections.
///
/// Input:  RGB texture 640x640
/// Output: List of Detection objects with class, confidence, bbox
///
/// This replaces the Python inference server for Quest 2 deployment.
/// No network connection needed — model runs entirely on-device.
/// </summary>
public class SentisDetector : MonoBehaviour
{
    [Header("Model")]
    public Unity.InferenceEngine.ModelAsset modelAsset;

    [Header("Camera")]
    public Camera perceptionCamera;
    public int inputWidth  = 640;
    public int inputHeight = 640;

    [Header("Detection Settings")]
    public float confidenceThreshold = 0.47f;
    public float iouThreshold        = 0.45f;
    public int   maxDetections       = 20;

    [Header("Class Names")]
    public string[] classNames = {
        "vehicle", "pedestrian",
        "traffic_sign", "traffic_light"
    };

    [Header("Class Colours")]
    public Color[] classColors = {
        Color.green,  // vehicle
        Color.red,    // pedestrian
        Color.yellow, // traffic_sign
        Color.blue    // traffic_light
    };

    // ── SENTIS ───────────────────────────────────────────
    private Unity.InferenceEngine.Model       runtimeModel;
    private Unity.InferenceEngine.Worker      worker;
    private RenderTexture inputRT;

    // ── DETECTION RESULT ─────────────────────────────────
    public class Detection
    {
        public int   classId;
        public string className;
        public float confidence;
        public Rect  screenRect;  // normalised 0-1
    }

    private List<Detection> latestDetections =
        new List<Detection>();
    private readonly object detectionLock = new object();

    // ── STATE ────────────────────────────────────────────
    private bool isInitialised = false;
    private int  frameCount    = 0;
    public  int  runEveryNFrames = 3;

    // Inference timing
    private float lastInferenceMs = 0f;

    void Start()
    {
        InitialiseSentis();
    }

    void InitialiseSentis()
    {
        if (modelAsset == null)
        {
            Debug.LogError(
                "[SentisDetector] No model asset assigned!");
            return;
        }

        // Load and compile the ONNX model
        // Sentis compiles it to optimised GPU instructions
        runtimeModel = Unity.InferenceEngine.ModelLoader.Load(modelAsset);

        // Create worker — this is the inference engine
        // BackendType.GPUCompute uses GPU shader compute
        // Falls back to CPU on unsupported hardware
        worker = new Unity.InferenceEngine.Worker(runtimeModel,
                            Unity.InferenceEngine.BackendType.GPUCompute);

        // Create input render texture
        // Same size as model input (640x640)
        inputRT = new RenderTexture(
            inputWidth, inputHeight, 0,
            RenderTextureFormat.ARGB32);

        // Attach to perception camera if assigned
        if (perceptionCamera != null)
            perceptionCamera.targetTexture = inputRT;

        isInitialised = true;
        Debug.Log("[SentisDetector] Initialised. " +
                  $"Model: {modelAsset.name} | " +
                  $"Backend: GPUCompute");
    }

    void Update()
    {
        if (!isInitialised) return;

        frameCount++;
        if (frameCount % runEveryNFrames == 0)
            StartCoroutine(RunInference());
    }

    IEnumerator RunInference()
    {
        yield return new WaitForEndOfFrame();

        if (inputRT == null || worker == null) yield break;

        float t = Time.realtimeSinceStartup;

        // ── PREPARE INPUT TENSOR ─────────────────────────
        // Convert RenderTexture to Sentis tensor
        // TextureTransform handles BGR->RGB and normalisation
        using (var inputTensor = Unity.InferenceEngine.TextureConverter.ToTensor(
            inputRT,
            inputWidth, inputHeight, 3))
        {
            // ── RUN INFERENCE ─────────────────────────────
            worker.Schedule(inputTensor);

            // ── READ OUTPUT ───────────────────────────────
            // Output shape: (1, 8, 8400)
            // Peek (not download) keeps tensor on GPU until needed
            var outputTensor = worker.PeekOutput("output0")
                as Unity.InferenceEngine.Tensor<float>;

            if (outputTensor == null) yield break;

            // Download to CPU for parsing
            // This is the GPU->CPU sync point
            var outputArray = outputTensor.DownloadToArray();

            // ── PARSE DETECTIONS ─────────────────────────
            var detections = ParseDetections(outputArray);

            // ── NMS ───────────────────────────────────────
            var filtered = ApplyNMS(detections);

            lock (detectionLock)
            {
                latestDetections = filtered;
            }

            lastInferenceMs =
                (Time.realtimeSinceStartup - t) * 1000f;
        }
    }

    List<Detection> ParseDetections(float[] output)
    {
        // Output tensor shape: (1, 8, 8400)
        // Flattened: [class0_scores..., class1_scores...,
        //             cx_values..., cy_values...,
        //             w_values..., h_values...]
        //
        // YOLOv8 output layout (transposed from training):
        // For each of 8400 candidates:
        //   indices [0..3]   = cx, cy, w, h (normalised)
        //   indices [4..7]   = class scores

        var detections = new List<Detection>();
        int numCandidates = 8400;
        int numValues     = 8; // 4 bbox + 4 classes

        for (int i = 0; i < numCandidates; i++)
        {
            // Extract bbox (normalised 0-1)
            float cx = output[0 * numCandidates + i];
            float cy = output[1 * numCandidates + i];
            float w  = output[2 * numCandidates + i];
            float h  = output[3 * numCandidates + i];

            // Find best class and its score
            int   bestClass = -1;
            float bestScore = 0f;

            for (int c = 0; c < classNames.Length; c++)
            {
                float score = output[
                    (4 + c) * numCandidates + i];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            // Filter by confidence
            if (bestScore < confidenceThreshold) continue;
            if (bestClass < 0) continue;

            // Convert cx,cy,w,h to x1,y1,x2,y2
            float x1 = cx - w / 2f;
            float y1 = cy - h / 2f;

            detections.Add(new Detection
            {
                classId    = bestClass,
                className  = classNames[bestClass],
                confidence = bestScore,
                screenRect = new Rect(x1, y1, w, h)
            });
        }

        return detections;
    }

    List<Detection> ApplyNMS(List<Detection> detections)
    {
        // Non-Maximum Suppression:
        // Remove overlapping boxes keeping only the
        // highest confidence detection per object.
        //
        // Sort by confidence descending
        detections.Sort((a, b) =>
            b.confidence.CompareTo(a.confidence));

        var kept = new List<Detection>();

        foreach (var det in detections)
        {
            if (kept.Count >= maxDetections) break;

            bool suppressed = false;
            foreach (var k in kept)
            {
                if (k.classId == det.classId &&
                    IoU(k.screenRect, det.screenRect)
                    > iouThreshold)
                {
                    suppressed = true;
                    break;
                }
            }

            if (!suppressed) kept.Add(det);
        }

        return kept;
    }

    float IoU(Rect a, Rect b)
    {
        // Intersection over Union
        float ix1 = Mathf.Max(a.xMin, b.xMin);
        float iy1 = Mathf.Max(a.yMin, b.yMin);
        float ix2 = Mathf.Min(a.xMax, b.xMax);
        float iy2 = Mathf.Min(a.yMax, b.yMax);

        if (ix2 <= ix1 || iy2 <= iy1) return 0f;

        float intersection = (ix2-ix1) * (iy2-iy1);
        float unionArea    = a.width * a.height +
                             b.width * b.height -
                             intersection;

        return unionArea > 0 ? intersection / unionArea : 0f;
    }

    // ── PUBLIC API ───────────────────────────────────────

    public List<Detection> GetDetections()
    {
        lock (detectionLock)
        {
            return new List<Detection>(latestDetections);
        }
    }

    public int GetDetectionCount()
    {
        lock (detectionLock)
        {
            return latestDetections.Count;
        }
    }

    public float GetLastInferenceMs() => lastInferenceMs;

    // ── CLEANUP ──────────────────────────────────────────

    void OnDestroy()
    {
        worker?.Dispose();
        if (inputRT != null) inputRT.Release();
    }

    // ── DEBUG GIZMOS ─────────────────────────────────────

void OnGUI()
{
    if (!Application.isPlaying) return;

    var detections = GetDetections();

    // Position on RIGHT side to avoid PerceptionBridge overlay
    float x = Screen.width - 320f;

    GUI.color = Color.cyan;
    GUI.Label(new Rect(x, 10, 310, 20),
        $"[Sentis] Det:{detections.Count} | " +
        $"Inf:{lastInferenceMs:F1}ms");

    int y = 35;
    foreach (var det in detections)
    {
        Color c = det.classId < classColors.Length
            ? classColors[det.classId] : Color.white;
        GUI.color = c;
        GUI.Label(new Rect(x, y, 310, 20),
            $"{det.className} {det.confidence:F2}");
        y += 20;
    }
}
}