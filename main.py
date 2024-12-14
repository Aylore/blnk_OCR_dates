import argparse
from src.train_model import train_model
from src.inference import predict
from utils.convert_to_onnx import convert_model_to_onnx
from utils.convert_to_trt import convert_model_to_trt
from utils.show_raw_data import show_raw_images
from config.config import data_path
from tensorflow import keras
import tensorflow as tf

# Enable eager execution and set logging level for TensorFlow
tf.compat.v1.enable_eager_execution()
tf.get_logger().setLevel('DEBUG')

def main(args):
    """
    Main function to handle different actions like showing data, training model, making predictions, or converting models.
    """

    if args.show_some_data:
        show_raw_images(args.data_path)
    elif args.train:
        train_model()
    elif args.image_path:
        model = keras.models.load_model(args.model_path)
        print(predict(model, args.image_path))
    elif args.to_onnx:
        convert_model_to_onnx(args.model_path)
    elif args.to_trt:
        convert_model_to_trt(args.onnx_model_path)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run model training, prediction, or conversion tasks.")
    
    parser.add_argument('--data-path', type=str, default=data_path, help='Path to the dataset')
    parser.add_argument('--show-some-data', action='store_true', help='Flag to show raw data images')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--image-path', type=str, help='Path to an image for inference')
    parser.add_argument('--model-path', type=str, help='Path to the trained model')
    parser.add_argument('--to-onnx', action='store_true', help='Flag to convert model to ONNX format')
    parser.add_argument('--to-trt', action='store_true', help='Flag to convert model to TensorRT format')
    parser.add_argument('--onnx-model-path', type=str, help='Path to the ONNX model for TRT conversion')

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args)
