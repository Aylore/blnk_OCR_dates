
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config.config import vocabulary

@tf.function
def map_chars():
    # Mapping characters to integers
    
    char_to_num = layers.StringLookup(
        vocabulary=vocabulary, mask_token=None
    )

    # Mapping integers back to original characters

    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
          mask_token=None,
            invert=True
    )


    return  char_to_num , num_to_char