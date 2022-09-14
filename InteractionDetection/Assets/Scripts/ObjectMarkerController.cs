using UnityEngine;
using YoloV4Tiny;
using TMPro;
using NRKernal;

// this class is used to define the behaviour of markers
// that track the position of detected objectes
public class ObjectMarkerController : MonoBehaviour
{
    // object label
    string target;
    Color myColor;
    GameObject center;

    enum MyColors{
        WHITE = 0,
        GREEN
    }

    Renderer myRenderer;

    TMP_Text label;

    int classIndx;

    bool audioPlayed;

    Vector3 topLeft, bottomRight;

    Camera RGBCamera;

    void Start(){
        myRenderer = gameObject.GetComponent<Renderer>();
        label = gameObject.GetComponentInChildren<TMP_Text>();
        RGBCamera = GameObject.FindGameObjectWithTag("RGBCamera").GetComponent<Camera>() as Camera;

        Hide();
    }

    public bool SetPosition(in Detection d){
        // get info from the bbox
        var x = d.x;
        //var y = (1-d.y);
        var y = d.y;
        var w = d.w;
        var h = d.h;

        Ray ray = RGBCamera.ViewportPointToRay(new Vector3(x, y));

        RaycastHit hitResult;
        if (Physics.Raycast(ray, out hitResult)){
            if (hitResult.collider.gameObject != null && hitResult.collider.gameObject.GetComponent<NRTrackableBehaviour>() != null)
            {
                var behaviour = hitResult.collider.gameObject.GetComponent<NRTrackableBehaviour>();

                // if the collider is a plane, set the position of the object on it
                if (behaviour.Trackable.GetTrackableType() == TrackableType.TRACKABLE_PLANE)
                {
                    transform.position = hitResult.point;
                    transform.rotation = Camera.main.transform.rotation;
                    return true;
                }
            }
        }
        return false;     
    }

    public bool Initialize(in Detection d){
        audioPlayed = false;

        // set the target
        classIndx = (int)d.classIndex;

        // set target label
        var targetLabel = Objects.labels[classIndx];
        if(!myRenderer.enabled)
            label.text = "";
        else
            label.text = targetLabel;

        if(SetPosition(d) && targetLabel != "hand"){
            Show();
            return true;
        }
        Hide();
        return false;
    }

    public bool GetAudioPlayed(){
        return audioPlayed;
    }

    public void SetAudioPlayed(bool played){
        audioPlayed = played;
    }

    public Vector3 GetCenterPos(){
        return transform.position;
    }

    // get the target's index associated to the marker
    public int GetTargetIndex(){
        return classIndx;
    }

    // set the color of the marker
    public void SetColor(Color color){
        myRenderer.material.color = color;
    }

    public Color GetColor(){
        return myRenderer.material.color;
    }

    public void SetLabel(string newLabel){
        label.text = newLabel;
    }

    public void Hide(){
        gameObject.SetActive(false);
    }

    public bool isActive(){
        return gameObject.activeSelf;
    }

    public void Show(){
        gameObject.SetActive(true);
    }
}
