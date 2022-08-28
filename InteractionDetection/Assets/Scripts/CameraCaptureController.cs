using UnityEngine;
using NRKernal;

// This class is used to capture and provide Nreal RGB camera frames
public class CameraCaptureController
{
    NRRGBCamTexture _RGBCamTexture;
    Texture _frameTexture;

    public CameraCaptureController()
    {
        _RGBCamTexture = new NRRGBCamTexture();
        _frameTexture = _RGBCamTexture.GetTexture();
        _RGBCamTexture.Play();
    }

    public Texture GetTexture(){
        return _frameTexture;
    }
}
