import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config.config import *


def decode_batch_predictions(pred):
    """
    Decode the output predictions of a neural network, typically used in Optical Character Recognition (OCR).

    Parameters:
    - pred (numpy.ndarray): The prediction output from the network.

    Returns:
    list: A list of decoded text predictions.

    This function takes the network's predictions and decodes them into human-readable text.
    It uses a greedy search approach, but for more complex tasks, beam search can be used.

    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # 5. Map the characters in label to numbers
    char_to_num = layers.StringLookup(
        vocabulary=vocabulary, mask_token=None
    )

    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )   
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text