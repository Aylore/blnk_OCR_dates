import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from config.config import img_height , img_width , epochs  ,vocabulary

from src.encode_imgs import map_chars

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it to the layer using `self.add_loss()`.
        
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time,return only the computed predictions
        return y_pred


def build_model():
    
    # Inputs to the model
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_uniform",padding="same",name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_uniform",padding="same",name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    # We have used two max pool with pool size and strides 2. So, downsampled feature maps are 4x smaller. 
    # The number of filters in the last layer is 64. Reshape accordingly before passing the output to the 
    # RNN part of the model
    
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    
    x = layers.Reshape(target_shape=new_shape, name="flattern")(x)
    
    
    x = layers.Dense(128, activation="relu", kernel_initializer="he_uniform", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    
    
    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)

    # Output layer
    # Mapping characters to integers
    
    char_to_num = layers.StringLookup(
        vocabulary=vocabulary, mask_token=None
    )

    # Mapping integers back to original characters

    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output)
    
    # Optimizer
    opt = keras.optimizers.Adam()
    
    # Compile the model and return
    model.compile(optimizer=opt)
    
    return model
