
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.decode_output import decode_batch_predictions
from src.preprocess import map_chars

import numpy as np
from utils.read_text import read_text_file
from src.encode_imgs import encode_single_sample

def predict(model , image):
    image = np.expand_dims(np.array(image) , 0)
    text = np.expand_dims(np.array(["random_text"]) , 0)

    infer_dataset = tf.data.Dataset.from_tensor_slices((image,  text
    ))
    infer_dataset = (
        infer_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(1)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )



    # _ , num_to_char = map_chars()
    for batch in infer_dataset:
        batch_images = batch["image"]
        # batch_labels = batch["label"]
        # batch_text = num_to_char(batch_labels)
        
        preds = model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)
        label_texts = []

        for i in range(len(preds)):
            # pred_i = pred_texts[i]
            # label_i = tf.strings.reduce_join(batch_text[i]).numpy().decode("utf-8")
            label_texts.append(pred_texts)


    return label_texts[0][0]
            