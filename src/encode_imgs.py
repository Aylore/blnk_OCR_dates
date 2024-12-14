import tensorflow as tf


from src.preprocess import map_chars
from tensorflow import keras
from tensorflow.keras import layers

from config.config import vocabulary


tf.compat.v1.enable_eager_execution()



@tf.function
def encode_single_sample(img_path, label):
    """
    Encodes a single image sample for training a machine learning model.

    Parameters:
    - img_path (str): The file path to the image.
    - label (str): The label or text associated with the image.

    Returns:
    dict: A dictionary containing the encoded image and label.

    Example:
    encode_single_sample("image.jpg", "١٩٩٩/٠٧/١٥")
    Output: {"image": <encoded_image_tensor>, "label": <encoded_label_tensor>}
    """
    
    # 1. Read the image
    img = tf.io.read_file(img_path)
    
    # 2. Decode and convert to grayscale
    img = tf.io.decode_jpeg(img,channels=1)
    
    # 3. Crop only the region of interest ROI
    img = tf.image.crop_to_bounding_box(img, 20, 150, 50,200)
    
    # 4. Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.image.rot90(img, k=1)
    
    # 5. Map the characters in label to numbers
    char_to_num = layers.StringLookup(
        vocabulary=vocabulary, mask_token=None
    )

    # Mapping integers back to original characters
    # num_to_char = layers.StringLookup(
    #     vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    # )    
    
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    
    # 6. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}