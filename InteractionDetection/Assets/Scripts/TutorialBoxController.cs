using System.Collections;
using UnityEngine;
using UnityEngine.Video;
using UnityEngine.UI;

public class TutorialBoxController : MonoBehaviour
{
    VideoPlayer _video;
    public RawImage _image;

    Time time;


    void Start(){
        _video = gameObject.GetComponent<VideoPlayer>();
    }

    IEnumerator PlayVideoTest(int index, string action){
        VideoClip clip = (VideoClip)Resources.Load($"Video/Test/{action}_alimentatore");
        
        switch(Objects.labels[index]){
            case "Alimentatore":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_alimentatore");
                break;
            case "Oscilloscopio":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_oscilloscopio");
                break;
            case "Stazione di saldatura":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_stazione_saldatura");
                break;
            case "Avvitatore elettrico":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_avvitatore_elettrico");
                break;
            case "Cacciavite":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_cacciavite");
                break;
            case "Pinza":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_pinza");
                break;
            case "Sonda oscilloscopio":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_sonda_oscilloscopio");
                break;
            case "Scheda a bassa tensione":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_scheda_bassa_tensione");
                break;
            case "Scheda ad alta tensione":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_scheda_alta_tensione");
                break;
            case "Batteria avvitatore":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_batteria_avvitatore");
                break;
            case "Area di lavoro":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_area_lavoro");
                break;
            case "Base saldatore":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_base_saldatore");
                break;
            case "Presa":
                clip = (VideoClip)Resources.Load($"Video/Test/{action}_presa");
                break;
        }
        // play the video clip
        _video.clip = clip;
        _video.Prepare();
        while(!_video.isPrepared){
            yield return new WaitForSeconds(1f);
        }
        _image.texture = _video.texture;
        _video.Play();
    }
    
    public void Show(int index, string action){
        StartCoroutine(PlayVideoTest(index, action));
    }

    public void Hide(){
        gameObject.SetActive(false);
    }

    public bool isActive(){
        return gameObject.activeSelf;
    }
}
