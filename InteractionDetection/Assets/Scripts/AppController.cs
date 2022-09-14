using System.Collections;
using UnityEngine;
using NRKernal;
using YoloV4Tiny;


public class AppController : MonoBehaviour
{
    [SerializeField] ResourceSet _resources = null;
    [SerializeField] ObjectMarkerController _markerPrefab = null;
    [SerializeField] InfoBoxController _infoPrefab = null;
    [SerializeField] ConfigController _config = null;
    [SerializeField, Range(0, 1)] float _scoreTreshold = 0.5f;

    AudioSource _audio;

    CameraCaptureController _capture;

    ObjectDetector _detector;

    HandInfo _hand;

    ObjectMarkerController[] _markers;

    InfoBoxController _info;

    // true if a the user is looking a plane surface
    bool _isLookingPlane = false;

    // offset needed for a better interaction detection
    //float _distanceThreshold = 0.2f;
    float _distanceThreshold = 0.05f;


    void Start()
    {
        _audio = gameObject.GetComponent<AudioSource>();
        _capture = new CameraCaptureController();
        _detector = new ObjectDetector(_resources);
        _markers = new ObjectMarkerController[10];
        _hand = new HandInfo();
        
        for (var i = 0; i < _markers.Length; i++){
            _markers[i] = Instantiate(_markerPrefab);
        }

        _info = Instantiate(_infoPrefab);

        // repeat the detection periodically
        StartCoroutine(DetectObjects());

    }

    void Update()
    {
        UpdateLookingStatus();
        if(_config.mode == ConfigController.InteractionMode.Hands)
            CheckHandsInteraction();
    }
    
    // Check if the user is looking a plane
    void UpdateLookingStatus(){
        var handControllerAnchor = NRInput.DomainHand == ControllerHandEnum.Left ? ControllerAnchorEnum.LeftLaserAnchor : ControllerAnchorEnum.RightLaserAnchor;
        Transform laserAnchor = NRInput.AnchorsHelper.GetAnchor(NRInput.RaycastMode == RaycastModeEnum.Gaze ? ControllerAnchorEnum.GazePoseTrackerAnchor : handControllerAnchor);
        // Check for a collision between a ray starting from the head and a plane
        RaycastHit hitResult;
        if (Physics.Raycast(new Ray(laserAnchor.transform.position, laserAnchor.transform.forward), out hitResult, 20))
        {
            if (hitResult.collider.gameObject != null && hitResult.collider.gameObject.GetComponent<NRTrackableBehaviour>() != null)
            {
                var behaviour = hitResult.collider.gameObject.GetComponent<NRTrackableBehaviour>();

                // if the collider is a plane, get its z and set isLookingPlane to true
                if (behaviour.Trackable.GetTrackableType() == TrackableType.TRACKABLE_PLANE)
                {
                    _isLookingPlane = true;
                    return;
                }
            }
        }
        _isLookingPlane = false;
    }


    bool CheckHandDistance(ObjectMarkerController marker, HandEnum hand){
        // get marker position
        var center = marker.GetCenterPos();

        //check if the hand is visible
        if(_hand.IsVisible(hand)){

            var palmPos = _hand.GetKeyPointPosition(hand, HandJointID.Palm);
            var thumbPos = _hand.GetKeyPointPosition(hand, HandJointID.ThumbTip);
            var indexPos = _hand.GetKeyPointPosition(hand, HandJointID.IndexTip);
            var middlePos = _hand.GetKeyPointPosition(hand, HandJointID.MiddleTip);
            var ringPos =_hand.GetKeyPointPosition(hand, HandJointID.RingTip);
            var pinkyPos =_hand.GetKeyPointPosition(hand, HandJointID.PinkyTip);

            if(Vector3.Distance(center,palmPos) <= _distanceThreshold) return true;
            if(Vector3.Distance(center,thumbPos) <= _distanceThreshold) return true;
            if(Vector3.Distance(center,indexPos) <= _distanceThreshold) return true;
            if(Vector3.Distance(center,middlePos) <= _distanceThreshold) return true;
            if(Vector3.Distance(center,ringPos) <= _distanceThreshold) return true;
            if(Vector3.Distance(center,pinkyPos) <= _distanceThreshold) return true;

        }
        return false;
    }

    //[COMING SOON]

    /*void CheckGazeInteraction(){
        var i = 0;
        foreach(var marker in _markers){
            if(marker.isActive()){
                var indx = marker.GetTargetIndex();
                var markerPos = marker.GetCenterPos();
                var distanceFromMarker = Vector3.Distance(markerPos, _hitPoint);
                if(distanceFromMarker < _distanceGazeThreshold){
                    marker.SetColor(Color.green);
                    _info.Show(indx, _displayTime);
                    if(!marker.GetAudioPlayed() && !_audio.isPlaying){
                        PlayAudio(indx);
                        marker.SetAudioPlayed(true);
                    }
                    i++;
                }else{
                    marker.SetColor(Color.white);
                    marker.SetAudioPlayed(false);
                }
            }
        }

        if(i==0)
            _info.Hide();
    }*/


    // check if the user is touching an object
    // with his left or right hand
    void CheckHandsInteraction(){
        var interacted = false;
        foreach(var marker in _markers){
            if(marker.isActive()){
                var indx = marker.GetTargetIndex();
                // if this marker is touched, get its label
                if((CheckHandDistance(marker, HandEnum.RightHand) || CheckHandDistance(marker, HandEnum.LeftHand)) && !interacted){
                    marker.SetColor(Color.green);
                    _info.Show(indx);
                    if(!marker.GetAudioPlayed()){
                        PlayAudio(indx);
                        marker.SetAudioPlayed(true);
                    }
                    interacted = true;
                }else{
                    marker.SetColor(Color.white);
                }
            }
        }
    }


    // detect objects in a frame's texture
    IEnumerator DetectObjects(){
        while(true){
            // start detecting only if the user is looking a plane 
            // and both hands are not visible (no intention to touch an object)
            if(_isLookingPlane && !(_hand.IsVisible(HandEnum.RightHand) || _hand.IsVisible(HandEnum.LeftHand)) && !_info.isActive()){
                var _frame = _capture.GetTexture();
                _detector.ProcessImage(_frame, _scoreTreshold);
                int i=0;
                int activeObj = 0;
                // set a marker for each detected object
                foreach(var d in _detector.Detections){
                    if (i == _markers.Length) break;

                    // show or hide the marker based on the toggle 'debug'
                    if(!_config.debug){
                        _markers[i].gameObject.GetComponent<Renderer>().enabled = false;
                    }else{
                         _markers[i].gameObject.GetComponent<Renderer>().enabled = true;
                    }

                    // set marker's position
                    var isActive = _markers[i].Initialize(d);
                    if(isActive)
                        activeObj++;

                    i++;
                }

                if(activeObj > 0)
                    _audio.PlayOneShot((AudioClip) Resources.Load("Audio/detection_completed"));

                // hide unused markers
                for(; i<_markers.Length; i++) _markers[i].Hide();
            }
            // do the next detection after 2 sec
            yield return new WaitForSeconds(2f);
        }
    }


    // set and play an audio clip based on the touched object label
    void PlayAudio(int indx){
        // default audio
        AudioClip clip = (AudioClip)Resources.Load("Audio/oggetto");

        switch(Objects.labels[indx]){
            case "Alimentatore":
                clip = (AudioClip)Resources.Load("Audio/alimentatore");
                break;
            case "Oscilloscopio":
                clip = (AudioClip)Resources.Load("Audio/oscilloscopio");
                break;
            case "Stazione di saldatura":
                clip = (AudioClip)Resources.Load("Audio/stazione_saldatura");
                break;
            case "Avvitatore elettrico":
                clip = (AudioClip)Resources.Load("Audio/avvitatore_elettrico");
                break;
            case "Cacciavite":
                clip = (AudioClip)Resources.Load("Audio/cacciavite");
                break;
            case "Pinza":
                clip = (AudioClip)Resources.Load("Audio/pinza");
                break;
            case "Punta del saldatore":
                clip = (AudioClip)Resources.Load("Audio/punta_saldatore");
                break;
            case "Sonda oscilloscopio":
                clip = (AudioClip)Resources.Load("Audio/sonda_oscilloscopio");
                break;
            case "Scheda a bassa tensione":
                clip = (AudioClip)Resources.Load("Audio/scheda_bassa_tensione");
                break;
            case "Scheda ad alta tensione":
                clip = (AudioClip)Resources.Load("Audio/scheda_alta_tensione");
                break;
            case "Batteria avvitatore":
                clip = (AudioClip)Resources.Load("Audio/batteria_avvitatore");
                break;
            case "Area di lavoro":
                clip = (AudioClip)Resources.Load("Audio/area_lavoro");
                break;
            case "Base saldatore":
                clip = (AudioClip)Resources.Load("Audio/base_saldatore");
                break;
            case "Presa":
                clip = (AudioClip)Resources.Load("Audio/presa");
                break;
        }
        // play the audio clip
        _audio.clip = clip;
        _audio.Play();
    }


    void OnDisable()
      => _detector.Dispose();

    void OnDestroy()
    {
        for (var i = 0; i < _markers.Length; i++) Destroy(_markers[i]);
        Destroy(_info);
    }
 
}
