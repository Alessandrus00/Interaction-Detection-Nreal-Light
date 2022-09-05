using UnityEngine;
using NRKernal;

// This class is used to get info about user's hand
public class HandInfo
{
    HandState GetState(HandEnum hand){
        return NRInput.Hands.GetHandState(hand);
    }

    public Vector3 GetKeyPointPosition(HandEnum hand, HandJointID joint){
        HandState handState = GetState(hand);
        Pose jointPose = handState.GetJointPose(joint);
        return jointPose.position;
    }

    // if the hand is in the FOV, then it is tracked
    public bool IsVisible(HandEnum hand){
        HandState handState = GetState(hand);
        return handState.isTracked;
    }
}
