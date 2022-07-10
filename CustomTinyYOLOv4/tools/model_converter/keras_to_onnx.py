import tensorflow as tf
import os
# Weight Quantization - Input/Output=float32
# INPUT  = input_1 (float32, 1 x 416 x 416 x 3)
# OUTPUT = conv2d_18, conv2d_21
os.environ['TF_KERAS'] = '1'
import keras2onnx

model = tf.keras.models.model_from_json(open('model_data/yolov4_tiny.json').read(), custom_objects={'tf': tf})
weights_path = input("model weights path: ")
model.load_weights(weights_path)
onnx_model = keras2onnx.convert_keras(model, model.name, channel_first_inputs=['input_1'], target_opset=9)
keras2onnx.save_model(onnx_model, 'model_data/yolov4_tiny.onnx')