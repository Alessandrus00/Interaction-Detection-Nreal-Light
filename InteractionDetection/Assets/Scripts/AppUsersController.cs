using System.Collections;
using UnityEngine;
using NRKernal;
using YoloV4Tiny;
using System;


public class AppUsersController : MonoBehaviour
{
    [SerializeField] ResourceSet _resources = null;
    [SerializeField] ObjectMarkerController _markerPrefab = null;
    [SerializeField] InfoBoxController _infoPrefab = null;
    [SerializeField] TutorialBoxController _tutorialPrefab = null;
    [SerializeField, Range(0, 1)] float _scoreTreshold = 0.5f;

    AudioSource _audio;

    CameraCaptureController _capture;

    ObjectDetector _detector;

    HandInfo _hand;

    ObjectMarkerController[] _markers;

    InfoBoxController _info;

    TutorialBoxController _tutorial;

    Vector3 _hitPoint;

    // true if a the user is looking a plane surface
    bool _isLookingPlane = false;

    // offset needed for a better interaction detection
    //float _distanceThreshold = 0.2f;
    float _distanceThreshold = 0.05f;

    int [] classIndexes;

    int completedSteps = 0;

    string action = "look";


    void Start()
    {
        _audio = gameObject.GetComponent<AudioSource>();
        _capture = new CameraCaptureController();
        _detector = new ObjectDetector(_resources);
        _markers = new ObjectMarkerController[10];
        _hand = new HandInfo();

        // classes to test
        classIndexes = new [] {
            Objects.index("Pinza"),
            Objects.index("Cacciavite"),
            Objects.index("Oscilloscopio"),
            Objects.index("Sonda oscilloscopio"),
            Objects.index("Base saldatore")
        };

        NRKernal.NRHandCapsuleVisual.showCapsule = NRKernal.NRHandCapsuleVisual.showJoint = false;
        NRKernal.NRExamples.PolygonPlaneVisualizer.visibility = false;

        
        for (var i = 0; i < _markers.Length; i++){
            _markers[i] = Instantiate(_markerPrefab);
            _markers[i].gameObject.GetComponent<Renderer>().enabled = false;
        }

        _info = Instantiate(_infoPrefab);

        _tutorial = Instantiate(_tutorialPrefab);

        StartCoroutine(NextStep());
    }

    Vector3 GetHitPointPosition(Transform laserAnchor){
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
                    return hitResult.point;
                }
            }
        }
        return Vector3.zero;
    }


    int Detect(){
        var _frame = _capture.GetTexture();
        _detector.ProcessImage(_frame, _scoreTreshold);
        int i=0;
        int activeObj = 0;
        // set a marker for each detected object
        foreach(var d in _detector.Detections){
            if (i == _markers.Length) break;
            // set marker's position
            var isActive = _markers[i].Initialize(d);
            if(isActive)
                activeObj++;

            i++;
        }

        // hide unused markers
        for(; i<_markers.Length; i++) _markers[i].Hide();
        
        return activeObj;
    }


    void PlayDetected(){
        AudioClip clip = (AudioClip) Resources.Load("Audio/detection_completed");
        _audio.PlayOneShot(clip);
    }

    bool CheckHandDistance(ObjectMarkerController marker, HandEnum hand){
        var center = marker.GetCenterPos();

        //check if the right hand is visible
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


    void Update(){
        if(completedSteps>=0 && completedSteps<=4)
            _tutorial.Show(classIndexes[completedSteps], action);
    }

    IEnumerator NextStep(){
        while(completedSteps < 5){
            var currClassIndex = classIndexes[completedSteps];
            
            yield return new WaitForSeconds(1f);
            
            // play audio with instructions
            AudioClip clip = GetAudioTestClip(currClassIndex);
            _audio.PlayOneShot(clip);

            // wait until audio clip stops
            while(_audio.isPlaying){
                yield return new WaitForSeconds(1f);
            }
            
            // check if is looking a plane and detect objects in the current frame
            while(true){
                Transform laserAnchor = NRInput.AnchorsHelper.GetAnchor(ControllerAnchorEnum.GazePoseTrackerAnchor);

                Vector3 point1 = GetHitPointPosition(laserAnchor);
                yield return new WaitForSeconds(2f);
                Vector3 point2 = GetHitPointPosition(laserAnchor);

                if(Vector3.Distance(point1, point2) < 2 && point1 != Vector3.zero && point2 != Vector3.zero){
                    var activeObj = Detect();
                    if(activeObj > 0){
                        PlayDetected();
                        break;
                    }
                }
                yield return new WaitForSeconds(1f);
            }

            yield return new WaitForSeconds(1.5f);
            
            action = "touch";
            PlayTouchObject();

            DateTime start = System.DateTime.Now;

            // check if there is an interaction with some objects
            var interactionDetected = false;
            while(!interactionDetected && System.DateTime.Now < start.AddSeconds(10)){
                foreach(var marker in _markers){
                    if(marker.isActive()){
                        var indx = marker.GetTargetIndex();
                        // if this marker is touched, get its label
                        if((CheckHandDistance(marker, HandEnum.RightHand) || CheckHandDistance(marker, HandEnum.LeftHand))){
                            _info.Show(indx);
                            if(!marker.GetAudioPlayed()){
                                AudioClip objClip = GetAudioClip(indx);
                                _audio.PlayOneShot(objClip);
                                marker.SetAudioPlayed(true);
                            }
                            interactionDetected = true;
                            break;
                        }
                    }
                }
                yield return new WaitForSeconds(1f);
            }

            // wait and move to the next step
            yield return new WaitForSeconds(10f);
            
            action = "look";
            completedSteps++;
        }

        _tutorial.Hide();

        PlayTestCompleted();

        yield return new WaitForSeconds(8);

        Application.Quit();
    }

    void PlayTestCompleted(){
        AudioClip clip = (AudioClip) Resources.Load("Audio/Test/test_completed");
        _audio.PlayOneShot(clip);
    }

    void PlayTouchObject(){
        AudioClip clip = (AudioClip) Resources.Load("Audio/Test/touch_object");
        _audio.PlayOneShot(clip);
    }

    AudioClip GetAudioClip(int indx){
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
        return clip;
    }

    AudioClip GetAudioTestClip(int indx){
        // default audio
        AudioClip clip = (AudioClip)Resources.Load("Audio/Test/punta_alimentatore");

        switch(Objects.labels[indx]){
            case "Alimentatore":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_alimentatore");
                break;
            case "Oscilloscopio":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_oscilloscopio");
                break;
            case "Stazione di saldatura":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_stazione_saldatura");
                break;
            case "Avvitatore elettrico":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_avvitatore_elettrico");
                break;
            case "Cacciavite":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_cacciavite");
                break;
            case "Pinza":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_pinza");
                break;
            case "Sonda oscilloscopio":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_sonda_oscilloscopio");
                break;
            case "Scheda a bassa tensione":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_scheda_bassa_tensione");
                break;
            case "Scheda ad alta tensione":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_scheda_alta_tensione");
                break;
            case "Batteria avvitatore":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_batteria_avvitatore");
                break;
            case "Area di lavoro":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_area_lavoro");
                break;
            case "Base saldatore":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_base_saldatore");
                break;
            case "Presa":
                clip = (AudioClip)Resources.Load("Audio/Test/punta_presa");
                break;
        }
        return clip;
    }


    void OnDisable()
      => _detector.Dispose();

    void OnDestroy()
    {
        for (var i = 0; i < _markers.Length; i++) Destroy(_markers[i]);
    }
 
}
