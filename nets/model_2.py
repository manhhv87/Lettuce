# Loading the library
import tensorflow as tf
from tensorflow.keras.layers import *


def Model_2(x):
    y = Flatten()(x)

    out_h2 = Dense(32, activation='relu')(y)
    out_h2 = Dense(64, activation='relu')(out_h2)
    out_h2 = Dense(1, name='h')(out_h2)

    out_d2 = Dense(32, activation='relu')(y)
    out_d2 = Dense(64, activation='relu')(out_d2)
    out_d2 = Dense(1, name='dia')(out_d2)

    return [out_h2, out_d2]
