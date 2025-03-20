import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import cv2
from nets.model_13 import *
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

##################### Parameter setting ##########################

# Original training data storage directory
f = '/home/manhhv87/AICode/TMSCNet_Lettuce/data/train/'

# Directory of label file corresponding to training data
f_json = f+'GroundTruth_SendJuly13.json'

shape = 64          # Size of the picture after scaling
loop_num = 100      # Multiple of image enhancement

batch_size = 500    # Number of batches of images loaded

units = 128         # Number of neurons in FCN
kernel_size = 3     # Convolution kernel size
pool_size = 2       # Pool core size
dropout = 0.1       # Dropout ratio
n_filters = 32      # Number of convolution kernels
layers = 1          # Layers of convolution kernel
n = 4               # Reduction times of neurons

learning_rate = 0.001   # Learning rate
epochs = 50             # Number of cycles

model_save_path = 'h5/model_13.h5'  # Model file storage path
file_name = 'Model_13'              # Training result storage file name

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Create folder
if os.path.isdir(f+'Debth_change') == False:
    os.mkdir(f+'Debth_change')
if os.path.isdir(f+'Debth_aug') == False:
    os.mkdir(f+'Debth_aug')

# Read in picture name
name = [p.split('/')[-1] for p in glob.glob(f+'Debth_*.png')]

# Image cropping and zooming
if os.path.isfile(f+'Debth_change/'+name[0]) == False:
    print('Start cropping and scaling the image!')
    [debth_color(name, f, shape) for name in name]
    print('Finish cropping and scaling.')

# Image enhancement
if os.path.isfile(f+'Debth_aug/'+name[0].split('.')[0]+'_0_1026.png') == False:
    print('Start data enhancement!')
    augment_depth(name, loop_num, f)
    print('Complete data enhancement.')

# Read in label file
index, fw, dw, h, dia, area, variety = load_json(f_json)

# Get the enhanced image path
aug_path = sorted(glob.glob(f+'Debth_aug/*.png'))[:loop_num]
aug_name = [p.split('/')[-1].split('Debth_100')[-1] for p in aug_path]

img_path = []
for i in range(len(index)):
    for j in range(loop_num):
        img_path.append(f+'Debth_aug/Debth_'+index[i]+aug_name[j])

# Label repeat loop_um times
h_ds = []
for i in range(len(index)):
    for j in range(loop_num):
        h_ds.append(h[i])
h_ds = np.array(h_ds)

# Pairing out of order
np.random.seed(116)
np.random.shuffle(img_path)
np.random.seed(116)
np.random.shuffle(h_ds)

# Making datasets
img_ds = tf.data.Dataset.from_tensor_slices(img_path).map(preprocess)
h_ds = tf.data.Dataset.from_tensor_slices(h_ds)

dataset = tf.data.Dataset.zip((img_ds, h_ds))
dataset = dataset.repeat().shuffle(100).batch(
    batch_size).prefetch(tf.data.experimental.AUTOTUNE)

n_test = int(len(img_path)*0.2)
n_train = len(img_path)-n_test
dataset_train = dataset.skip(n_test)
dataset_test = dataset.take(n_train)

# Establish network
inputs = Input(shape=(shape, shape, 3))
outputs = Model_13(inputs, n_filters, layers, kernel_size,
                   pool_size, dropout, units, n)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse'
    )

# Training model
print('There is a long data processing time before the training starts, please wait patiently!')
history = model.fit(dataset_train,
                    steps_per_epoch=n_train//batch_size,
                    epochs=epochs,
                    validation_data=dataset_test,
                    validation_steps=n_test//batch_size,
                    shuffle=True
                    )

# Save model
model.save(model_save_path)

# Print model
model.summary()

# Store training parameters
if not os.path.exists('result/{}'.format(file_name)):
    os.makedirs('result/{}'.format(file_name))

np.save('result/{}/train_loss_h.npy'.format(file_name), history.history['loss'])
np.save('result/{}/test_loss_h.npy'.format(file_name),  history.history['val_loss'])
