#--------------------------------------------#
# This part of the code is used to see the network structure
#--------------------------------------------#
from nets.yolo import yolo_body

if __name__ == "__main__":
    input_shape     = (416, 416, 3)
    anchors_mask    = [[3, 4, 5], [1, 2, 3]]
    num_classes     = 20

    model = yolo_body(input_shape, anchors_mask, num_classes, phi = 0)
    model.summary()
    json_string = model.to_json()
    open('model_data/yolov4_tiny.json', 'w').write(json_string)
