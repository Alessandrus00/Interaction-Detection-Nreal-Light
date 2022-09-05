using UnityEngine;

public class ConfigController : MonoBehaviour
{
    public enum InteractionMode
    {
        Hands,
        Gaze
    }

    public InteractionMode mode;
    public bool debug = true;
    public bool visibleHands = true;
    public bool visiblePlane = true;
    public bool tutorial = false;
}
