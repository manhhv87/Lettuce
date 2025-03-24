import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from nets.model_2 import *
from sklearn.preprocessing import StandardScaler
from dataset.read_label import *
from dataset.data_preprocess import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import *
from PIL import Image
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

##################### Parameter setting ）##########################

# Original training data storage directory
f = '/home/manhhv87/AICode/TMSCNet_Lettuce/data/train/'

# Directory of label file corresponding to training data
f_json = f+'GroundTruth_SendJuly13.json'

batch_size = 16         # Number of batches of images loaded

validation_split = 0.2  # Partition ratio of training set and verification set

learning_rate = 0.001   # Learning rate
epochs = 200            # Number of cycles

model_save_path = 'h5/model_2.h5'  # Model file storage path

###########################################################################

# Read in label file
index, fw, dw, h, dia, area, variety = load_json(f_json)
var_ds, dia_ds, h_ds, dw_ds, fw_ds, area_ds = np.array(variety), np.array(
    dia), np.array(h), np.array(dw), np.array(fw), np.array(area)

# Making datasets
x = np.zeros((len(variety), 2))
y_h = np.zeros(len(variety))
y_dia = np.zeros(len(variety))

for i in range(len(variety)):
    x[i, 0] = dw_ds[i]
    x[i, 1] = var_ds[i]
    y_h[i] = h_ds[i]
    y_dia[i] = dia_ds[i]

x = StandardScaler().fit_transform(x)

print(x)

# Pairing out of order
np.random.seed(116)
np.random.shuffle(x)
np.random.seed(116)
np.random.shuffle(y_h)
np.random.seed(116)
np.random.shuffle(y_dia)

# Establish network
inputs = Input(shape=(2,))
model = tf.keras.Model(inputs=inputs, outputs=Model_2(inputs))

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse'
    )

# Training model
model.fit(x, {'h': y_h, 'dia': y_dia},
          shuffle=True,
          epochs=epochs, batch_size=batch_size,
          validation_split=validation_split)

# Save model
model.save(model_save_path)

# Print model
model.summary()
