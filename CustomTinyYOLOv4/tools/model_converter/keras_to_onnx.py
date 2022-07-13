import tensorflow as tf
import os

os.environ['TF_KERAS'] = '1'
import keras2onnx

model_file = open('model_data/yolov4_tiny.json').read()
model = tf.keras.models.model_from_json(model_file, custom_objects={'tf': tf})
weights_path = input('model weights path: ')
model.load_weights(weights_path)
onnx_model = keras2onnx.convert_keras(
    model, 
    model.name, 
    channel_first_inputs=['input_1'], 
    target_opset=9
    )
keras2onnx.save_model(onnx_model, 'model_data/yolov4_tiny.onnx')