
from src.model import build_model
from tensorflow import keras

from src.dataset import prepare_dataset

from src.split_data import split_data

from config.config import epochs , data_path

from src.load_data import load_data

# # Get the model
#
def train_model():

    model = build_model()
    print(model.summary())

    early_stopping_patience = 7

    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping( 
        monitor="val_loss", 
        patience=early_stopping_patience, 
        restore_best_weights=True
    )
    print("load data")
    images , labels  = load_data(data_path)
    print("split data")
    x_train , x_valid , y_train , y_valid = split_data(images , labels)
    print("initiate prepare dataset")
    train_dataset ,validation_dataset = prepare_dataset(x_train , x_valid , y_train , y_valid)

    # Train the model
    print("training")
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],

    )
    model.save("models/model.h5")