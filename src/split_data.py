from sklearn.model_selection import train_test_split

import numpy as np


def split_data(images , labels , test_size = 0.25 ):

    # Split the data into train and val sets

    x_train, x_valid, y_train, y_valid = train_test_split(np.array(images),
                                                           np.array(labels),
                                                           test_size=test_size,
                                                           random_state=3407)

    return  x_train, x_valid, y_train, y_valid