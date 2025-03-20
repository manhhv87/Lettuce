import tensorflow as tf
from tensorflow.keras.layers import *


def Model_3(x):
    y = Flatten()(x)

    out_fw2 = Dense(32, activation='relu')(y)
    out_fw2 = Dense(64, activation='relu')(out_fw2)
    out_fw2 = Dense(1, name='fw')(out_fw2)

    out_la2 = Dense(32, activation='relu')(y)
    out_la2 = Dense(64, activation='relu')(out_la2)
    out_la2 = Dense(1, name='area')(out_la2)

    return [out_fw2, out_la2]
