import onnx

model = onnx.load("yolov8n.onnx")
graph = model.graph

# List all outputs
for output in graph.output:
    shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
    print(output.name, shape)
