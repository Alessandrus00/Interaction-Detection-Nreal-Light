using UnityEngine;
using TMPro;
using UnityEngine.Video;
using UnityEngine.UI;
using System.Collections;
using System;

public class InfoBoxController : MonoBehaviour
{
    [SerializeField] TMP_Text _label;
    [SerializeField] TMP_Text _description;
    VideoPlayer _video;
    public RawImage _image;

    Time time;

    void Start(){
        _video = gameObject.GetComponent<VideoPlayer>();
        Hide();
    }

    IEnumerator PlayVideo(int indx){
        VideoClip clip = (VideoClip)Resources.Load("Video/alimentatore");
        
        switch(Objects.labels[indx]){
            case "Alimentatore":
                clip = (VideoClip)Resources.Load("Video/alimentatore");
                break;
            case "Oscilloscopio":
                clip = (VideoClip)Resources.Load("Video/oscilloscopio");
                break;
            case "Stazione di saldatura":
                clip = (VideoClip)Resources.Load("Video/stazione_saldatura");
                break;
            case "Avvitatore elettrico":
                clip = (VideoClip)Resources.Load("Video/avvitatore_elettrico");
                break;
            case "Cacciavite":
                clip = (VideoClip)Resources.Load("Video/cacciavite");
                break;
            case "Pinza":
                clip = (VideoClip)Resources.Load("Video/pinza");
                break;
            case "Punta del saldatore":
                clip = (VideoClip)Resources.Load("Video/base_saldatore");
                break;
            case "Sonda oscilloscopio":
                clip = (VideoClip)Resources.Load("Video/sonda_oscilloscopio");
                break;
            case "Scheda a bassa tensione":
                clip = (VideoClip)Resources.Load("Video/scheda_bassa_tensione");
                break;
            case "Scheda ad alta tensione":
                clip = (VideoClip)Resources.Load("Video/scheda_alta_tensione");
                break;
            case "Batteria avvitatore":
                clip = (VideoClip)Resources.Load("Video/batteria_avvitatore");
                break;
            case "Area di lavoro":
                clip = (VideoClip)Resources.Load("Video/area_lavoro");
                break;
            case "Base saldatore":
                clip = (VideoClip)Resources.Load("Video/base_saldatore");
                break;
            case "Presa":
                clip = (VideoClip)Resources.Load("Video/presa");
                break;
        }
        // play the video clip
        _video.clip = clip;
        _video.Prepare();
        while(!_video.isPrepared){
            yield return new WaitForSeconds(0.5f);
        }
        _image.texture = _video.texture;
        _video.Play();

        DateTime start = System.DateTime.Now;

        while(_video.isPlaying && System.DateTime.Now < start.AddSeconds(5)){
            yield return new WaitForSeconds(0.5f);
        }
        Hide();
    }
    
    public void Show(int classIndx){
        if(isActive())
            return;
        gameObject.SetActive(true);
        _label.text = Objects.labels[classIndx];
        _description.text = Objects.descriptions[classIndx];
        StartCoroutine(PlayVideo(classIndx));
    }

    public void Hide(){
        gameObject.SetActive(false);
    }

    public bool isActive(){
        return gameObject.activeSelf;
    }
}
