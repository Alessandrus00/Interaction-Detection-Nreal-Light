# Training a custom Tiny YOLOv4 model with Keras

In this section a custom model of Tiny YOLOv4 is trained using Keras framework. Training dataset used in this project is available <a href="https://iplab.dmi.unict.it/EHOI_SYNTH">here</a>, under **Download Real Dataset** (you just need **Annotations** and **Images**); you could also use your own dataset. Once downloaded, you have to convert it in PASCAL VOC format (you can follow <a href="https://roboflow.com/convert/coco-json-to-pascal-voc-xml"> this tutorial</a> made by **Roboflow**). After that just copy the **.xml** and **.jpg** into `VOCdevkit/VOC2007/Annotations/` and `VOCdevkit/VOC2007/JPEGImages/` respectively. Note that in Roboflow you don't have to split the dataset, it will be done later. Next steps are:
1. Create a conda environment and install the requirements in `requirements.txt` via **pip**;
2. run `voc_annotations.py` to split the dataset in **training**, **validation** and **test** sets and convert them to YOLO format (see `2007_train.txt` and `2007_val.txt` created after this script);
3. run `train.py` to start training with your GPU (if you want to use the CPU just install tensorflow 1.14 instead of tensorflow-gpu 1.14);
4. run `summary.py` to create a **json** file that contains the structure of the network;
5. run `yolo.py` and `predict.py` to test out your model (by default it uses the last added weights in `logs`);
6. (optional) evaluate the model with `get_map.py`.

If you want to use this model in barracuda do the following extra steps:

7. run `tools/model_converter/keras_to_onnx.py` to convert Keras weights (**.h5**) + network structure (**.json**) to ONNX format;
8. run `tools/model_converter/onnx_to_barracuda.py` to get a barracuda compatible version of the model.

Note that inside `logs` and `model_data` there are files already created by me executing the previous steps on this dataset. You can delete them if you want to restart the all process or change dataset (in `model_data` keep only `simhei.ttf`, `yolo_anchors.txt` and `yolov4_tiny_weights_coco.h5`).

# Tensorflow on MAC M1 chip
To execute Tensorflow on new M1 chips, check `tf-mac-m1`, which contains a small guide to make it work.

# References
This repository was originally created by <a href="https://github.com/bubbliiiing/yolov4-tiny-keras"> Bubbliiiing</a> . I just made few changes to pursue the goal of this repo. The code inside `onnx_to_barracuda.py` was provided by <a href="https://github.com/keijiro/YoloV4TinyBarracuda">keijiro</a> through <a href="https://colab.research.google.com/drive/1YjSQ0IJvKimrc5-I4QXaWJ43-nbPqKOS?usp=sharing"> this colab notebook</a>.


