# Convert ONNX to TensorRT
import tf2onnx

def convert_model_to_onnx(model_path):

    onnx_model, _ = tf2onnx.convert.from_keras(model_path)

    with open('models/model.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())



