using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class ConfigPanelController : MonoBehaviour
{
    [SerializeField]
    private ConfigController m_ConfigController;
    [SerializeField]
    private Dropdown m_ModeDropDown;
    [SerializeField]
    private Toggle m_DebugToggle;
    [SerializeField]
    private Toggle m_HandsToggle;
    [SerializeField]
    private Toggle m_PlaneToggle;
    [SerializeField]
    private Toggle m_TutorialToggle;

    List<string> _ModeOptions = new List<string>() {
        ConfigController.InteractionMode.Hands.ToString(),
        ConfigController.InteractionMode.Gaze.ToString()
    };

    void Start()
    {
        InitConfigPanel();
    }

    private void InitConfigPanel()
    {
        m_ModeDropDown.options.Clear();
        m_ModeDropDown.AddOptions(_ModeOptions);
        int default_quality_index = 0;
        for (int i = 0; i < _ModeOptions.Count; i++)
        {
            if (_ModeOptions[i].Equals(m_ConfigController.mode.ToString()))
            {
                default_quality_index = i;
            }
        }
        m_ModeDropDown.value = default_quality_index;
        m_ModeDropDown.onValueChanged.AddListener((index) =>
        {
            Enum.TryParse<ConfigController.InteractionMode>(_ModeOptions[index],
                out m_ConfigController.mode);
        });

        m_DebugToggle.isOn = m_ConfigController.debug;
        if(!m_DebugToggle.isOn)
            m_ConfigController.debug = false;

        m_DebugToggle.onValueChanged.AddListener((val) =>
        {
            m_ConfigController.debug = val;
        });

        m_HandsToggle.isOn = m_ConfigController.visibleHands;
        if(!m_HandsToggle.isOn){
            NRKernal.NRHandCapsuleVisual.showCapsule = NRKernal.NRHandCapsuleVisual.showJoint = false;
            m_ConfigController.visibleHands = false;
        }

        m_HandsToggle.onValueChanged.AddListener((val) =>
        {
            NRKernal.NRHandCapsuleVisual.showCapsule = NRKernal.NRHandCapsuleVisual.showJoint = val;
            m_ConfigController.visibleHands = val;
        });

        m_PlaneToggle.isOn = m_ConfigController.visiblePlane;
        if(!m_PlaneToggle.isOn){
            NRKernal.NRExamples.PolygonPlaneVisualizer.visibility = false;
            m_ConfigController.visiblePlane = false;
        }
        m_PlaneToggle.onValueChanged.AddListener((val) =>
        {
            NRKernal.NRExamples.PolygonPlaneVisualizer.visibility = val;
            m_ConfigController.visiblePlane = val;
        });

        m_TutorialToggle.isOn = m_ConfigController.tutorial;
        if(m_TutorialToggle.isOn){
            SceneManager.LoadScene("MySceneTest");
            m_ConfigController.tutorial = true;
        }
        m_TutorialToggle.onValueChanged.AddListener((val) =>
        {
            SceneManager.LoadScene("MySceneTest");
            m_ConfigController.tutorial = val;
        });
    }
}
