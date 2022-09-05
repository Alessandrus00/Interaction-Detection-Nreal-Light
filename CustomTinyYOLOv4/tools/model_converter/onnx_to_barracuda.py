import numpy as np
import onnx
from onnx import checker, helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import numpy_helper as np_helper

old_onnx = 'model_data/yolov4_tiny.onnx'
new_onnx = 'model_data/yolov4_tiny_barracuda.onnx'


def scan_split_ops(model):
  for i in range(len(model.graph.node)):
    # Node type check
    node = model.graph.node[i]
    if node.op_type != 'Split': continue
    # Output tensor shape
    output = next(v for v in model.graph.value_info if v.name == node.output[0])
    shape = tuple(map(lambda x: x.dim_value, output.type.tensor_type.shape.dim))
    shape = (shape[3], shape[3])
    # "split" attribute addition
    new_node = helper.make_node('Split', node.input, node.output, split = shape, axis = 3)
    # Node replacement
    model.graph.node.insert(i, new_node)
    model.graph.node.remove(node)

model = onnx.load(old_onnx)
model = onnx.shape_inference.infer_shapes(model)
scan_split_ops(model)
checker.check_model(model)
onnx.save(model,new_onnx)