import onnx
# import onnx_tensorrt.backend as backend

def convert_model_to_trt(model_path):
    # Load your ONNX model
    onnx_model = onnx.load(model_path)

    # Convert ONNX model to TensorRT engine
    engine = backend.prepare(onnx_model, device='CUDA:0')

    # Save the engine to a file
    with open("models/model.trt", "wb") as f:
        f.write(engine.serialize())