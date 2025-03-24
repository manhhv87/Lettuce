import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from nets.model_12 import *
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

##################### Parameter setting #########################

# Original training data storage directory
f = '/home/manhhv87/AICode/TMSCNet_Lettuce/data/train/'

# Directory of label file corresponding to training data
f_json = f+'GroundTruth_SendJuly13.json'

box = [750, 250, 1350, 850]     # Crop the box of the original image
shape = 64                      # Size of the picture after scaling
loop_num = 100                  # Multiple of image enhancement

batch_size = 32             # Number of batches of images loaded
validation_split = 0.2      # Partition ratio of training set and verification set

learning_rate = 0.001       # Learning rate
epochs = 50                 # Number of cycles

model_save_path = 'h5/model_12.h5'  # Model file storage path
file_name = 'Model_12'              # Training result storage file name

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Create folder
if os.path.isdir(f+'RGB_change') == False:
    os.mkdir(f+'RGB_change')
if os.path.isdir(f+'RGB_aug') == False:
    os.mkdir(f+'RGB_aug')

# Read in picture name
name = [p.split('/')[-1] for p in glob.glob(f+'RGB_*.png')]

# Image cropping and zooming
if os.path.isfile(f+'RGB_change/'+name[0]) == False:
    print('Start cropping and scaling the image!')
    [crop_resize(name, f, box, shape) for name in name]
    print('Finish cropping and scaling.')

# Image enhancement
if os.path.isfile(f+'RGB_aug/'+name[0].split('.')[0]+'_0_1026.png') == False:
    print('Start data enhancement!')
    augment(name, loop_num, f)
    print('Complete data enhancement.')

# Read in label file
index, fw, dw, h, dia, area, variety = load_json(f_json)


# Get the enhanced image path
aug_path = sorted(glob.glob(f+'RGB_aug/*.png'))[:loop_num]
aug_name = [p.split('/')[-1].split('RGB_100')[-1] for p in aug_path]

img_path = []
for i in range(len(index)):
    for j in range(loop_num):
        img_path.append(f+'RGB_aug/RGB_'+index[i]+aug_name[j])

# Label repeat loop_ Num times
var_ds, fw_ds, dw_ds, h_ds, dia_ds, area_ds = [], [], [], [], [], []
for i in range(len(index)):
    for j in range(loop_num):
        var_ds.append(variety[i])

var_ds = np.array(var_ds)

# Pairing out of order
np.random.seed(116)
np.random.shuffle(img_path)
np.random.seed(116)
np.random.shuffle(var_ds)

# Making datasets （
img_ds = tf.data.Dataset.from_tensor_slices(img_path).map(preprocess)
var_ds = tf.data.Dataset.from_tensor_slices(var_ds)

dataset = tf.data.Dataset.zip((img_ds, var_ds))
dataset = dataset.repeat().shuffle(100).batch(
    batch_size).prefetch(tf.data.experimental.AUTOTUNE)

n_test = int(len(img_path)*validation_split)
n_train = len(img_path)-n_test
dataset_train = dataset.skip(n_test)
dataset_test = dataset.take(n_train)

# Establish network
inputs = Input(shape=(shape, shape, 3))
outputs = Model_12(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
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
    
np.save('result/{}/train_loss.npy'.format(file_name), history.history['loss'])
np.save('result/{}/val_loss.npy'.format(file_name),   history.history['val_loss'])
np.save('result/{}/train_acc.npy'.format(file_name),  history.history['sparse_categorical_accuracy'])
np.save('result/{}/val_acc.npy'.format(file_name),    history.history['val_sparse_categorical_accuracy'])
