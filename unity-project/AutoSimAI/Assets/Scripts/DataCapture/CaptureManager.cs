using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class CaptureManager : MonoBehaviour
{
    [Header("Capture Settings")]
    public int captureEveryNFrames = 10;
    public string outputFolder = "CaptureData";
    public Camera rgbCamera;
    public Camera depthCamera;
    public Camera segmentationCamera;

    [Header("Label Configuration")]
    public List<LabelEntry> labels = new List<LabelEntry>();

    private int frameCount = 0;
    private int captureIndex = 0;
    private string sessionFolder;

    // Render textures
    private RenderTexture rgbRT;
    private RenderTexture depthRT;
    private RenderTexture segRT;

    [Serializable]
    public class LabelEntry
    {
        public string className;
        public Color segmentationColor;
    }

    void Start()
    {
        Debug.Log($"[CaptureManager] Saving data to: {sessionFolder}");
        // Create output folder with timestamp
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        sessionFolder = Path.Combine(Application.dataPath, "..", outputFolder, timestamp);
        Directory.CreateDirectory(Path.Combine(sessionFolder, "rgb"));
        Directory.CreateDirectory(Path.Combine(sessionFolder, "depth"));
        Directory.CreateDirectory(Path.Combine(sessionFolder, "segmentation"));
        Directory.CreateDirectory(Path.Combine(sessionFolder, "labels"));

        Debug.Log($"[CaptureManager] Saving data to: {sessionFolder}");

        // Set up render textures
        rgbRT = new RenderTexture(1920, 1080, 24, RenderTextureFormat.ARGB32);
        depthRT = new RenderTexture(1920, 1080, 24, RenderTextureFormat.Depth);
        segRT = new RenderTexture(1920, 1080, 24, RenderTextureFormat.ARGB32);

        if (rgbCamera != null) rgbCamera.targetTexture = rgbRT;
        if (depthCamera != null) depthCamera.targetTexture = depthRT;
        if (segmentationCamera != null) segmentationCamera.targetTexture = segRT;
    }

    void Update()
    {
        frameCount++;
        if (frameCount % captureEveryNFrames == 0)
        {
            StartCoroutine(CaptureFrame());
        }
    }

    System.Collections.IEnumerator CaptureFrame()
    {
        // Wait for end of frame so rendering is complete
        yield return new WaitForEndOfFrame();

        string index = captureIndex.ToString("D6");

        // Save RGB
        SaveRenderTexture(rgbRT, 
            Path.Combine(sessionFolder, "rgb", $"rgb_{index}.jpg"), 
            TextureFormat.RGB24, true);

        // Save Depth
        SaveDepthTexture(depthRT,
            Path.Combine(sessionFolder, "depth", $"depth_{index}.png"));

        // Save bounding box labels
        SaveBoundingBoxes(index);

        captureIndex++;
    }

    void SaveRenderTexture(RenderTexture rt, string path, 
        TextureFormat format, bool jpg)
    {
        if (rt == null) return;

        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(rt.width, rt.height, format, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();

        byte[] bytes = jpg ? tex.EncodeToJPG(90) : tex.EncodeToPNG();
        File.WriteAllBytes(path, bytes);

        Destroy(tex);
        RenderTexture.active = null;
    }

    void SaveDepthTexture(RenderTexture rt, string path)
    {
        if (rt == null) return;

        // Create a temporary ARGB texture to read depth data
        RenderTexture tempRT = new RenderTexture(rt.width, rt.height, 0, 
            RenderTextureFormat.ARGB32);
        Graphics.Blit(rt, tempRT);

        RenderTexture.active = tempRT;
        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();

        byte[] bytes = tex.EncodeToPNG();
        File.WriteAllBytes(path, bytes);

        Destroy(tex);
        Destroy(tempRT);
        RenderTexture.active = null;
    }

    void SaveBoundingBoxes(string index)
    {
        if (rgbCamera == null) return;

        List<string> annotations = new List<string>();
        
        // Find all labelled objects in scene
        LabelledObject[] labelledObjects = 
            FindObjectsByType<LabelledObject>(FindObjectsSortMode.None);

        int imgW = rgbRT.width;
        int imgH = rgbRT.height;

        foreach (LabelledObject obj in labelledObjects)
        {
            // Get renderer bounds
            Renderer rend = obj.GetComponentInChildren<Renderer>();
            if (rend == null) continue;

            // Check if object is in front of camera
            Vector3 viewPos = rgbCamera.WorldToViewportPoint(
                rend.bounds.center);
            if (viewPos.z < 0) continue; // Behind camera
            if (viewPos.x < 0 || viewPos.x > 1 || 
                viewPos.y < 0 || viewPos.y > 1) continue; // Off screen

            // Project bounding box corners to screen space
            Bounds bounds = rend.bounds;
            Vector3[] corners = GetBoundsCorners(bounds);

            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;
            bool anyVisible = false;

            foreach (Vector3 corner in corners)
            {
                Vector3 screenPos = rgbCamera.WorldToScreenPoint(corner);
                if (screenPos.z > 0)
                {
                    minX = Mathf.Min(minX, screenPos.x);
                    maxX = Mathf.Max(maxX, screenPos.x);
                    minY = Mathf.Min(minY, screenPos.y);
                    maxY = Mathf.Max(maxY, screenPos.y);
                    anyVisible = true;
                }
            }

            if (!anyVisible) continue;

            // Clamp to screen
            minX = Mathf.Clamp(minX, 0, imgW);
            maxX = Mathf.Clamp(maxX, 0, imgW);
            minY = Mathf.Clamp(minY, 0, imgH);
            maxY = Mathf.Clamp(maxY, 0, imgH);

            // Convert to YOLO format
            // cx, cy, w, h all normalised 0-1
            float cx = ((minX + maxX) / 2f) / imgW;
            float cy = 1f - ((minY + maxY) / 2f) / imgH; // Flip Y
            float w = (maxX - minX) / imgW;
            float h = (maxY - minY) / imgH;

            // Skip tiny boxes
            if (w < 0.01f || h < 0.01f) continue;

            int classId = obj.classId;
            annotations.Add($"{classId} {cx:F6} {cy:F6} {w:F6} {h:F6}");
        }

        // Save YOLO format label file
        string labelPath = Path.Combine(sessionFolder, "labels", 
            $"rgb_{index}.txt");
        File.WriteAllLines(labelPath, annotations);
    }

    Vector3[] GetBoundsCorners(Bounds b)
    {
        return new Vector3[]
        {
            new Vector3(b.min.x, b.min.y, b.min.z),
            new Vector3(b.min.x, b.min.y, b.max.z),
            new Vector3(b.min.x, b.max.y, b.min.z),
            new Vector3(b.min.x, b.max.y, b.max.z),
            new Vector3(b.max.x, b.min.y, b.min.z),
            new Vector3(b.max.x, b.min.y, b.max.z),
            new Vector3(b.max.x, b.max.y, b.min.z),
            new Vector3(b.max.x, b.max.y, b.max.z),
        };
    }

    void OnDestroy()
    {
        if (rgbRT != null) rgbRT.Release();
        if (depthRT != null) depthRT.Release();
        if (segRT != null) segRT.Release();
    }
}