import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# from src.split_data import split_data

from config.config import batch_size

from src.encode_imgs import encode_single_sample

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

tf.get_logger().setLevel('DEBUG')

@tf.function
def prepare_dataset(x_train , x_valid , y_train , y_valid):
    if not y_valid :
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds = (
            ds.map(
                encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return ds
    # Prepare train dataset
    print("prepare train")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Prepare validation dataset
    print("prepare validation")
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


    return train_dataset , validation_dataset