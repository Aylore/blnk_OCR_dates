import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from config.config import vocabulary
from src.load_data import load_data
import numpy as np

from src.dataset import prepare_dataset

def show_raw_images(data_path):
    images , labels  = load_data(data_path)


    data_set  = prepare_dataset(x_train = np.array(images) ,y_train = np.array(labels),
                                     x_valid = None , y_valid = None)


    
    char_to_num = layers.StringLookup(
        vocabulary=vocabulary, mask_token=None
    )

    # Mapping integers back to original characters

    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        mask_token=None,
            invert=True
    )


    _, ax = plt.subplots(4, 4, figsize=(18, 8))

    for batch in data_set.take(1):
        images = batch["image"]
        labels = batch["label"]
        for i in range(16):
            img = images[i].numpy().astype("uint8")
            img = tf.image.rot90(img, k=3)
            label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
            ax[i // 4, i % 4].imshow(img[:, :, 0], cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")
    plt.show()