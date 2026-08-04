using UnityEngine;

public class LabelledObject : MonoBehaviour
{
    [Tooltip("Class ID matching your data.yaml: 0=vehicle, 1=pedestrian, 2=traffic_sign, 3=traffic_light")]
    public int classId = 0;

    [Tooltip("Human readable class name")]
    public string className = "vehicle";
}