# Nreal Light application

This android application has been developed with Unity and uses the NRSDK library to access Nreal Light glasses hardware. It's purpose is to detect interactions between the hand of the user (through **hand tracking**) and some objects detected by the object detector **Tiny YOLOv4** (trained with custom data). Once an interaction is detected, the application shows some information about the touched object, to enrich the knowledge of the user.

# References

The trained model was used in Unity through Baracuda, thanks to the package provided by <a href="https://github.com/keijiro/YoloV4TinyBarracuda">keijiro</a>.
