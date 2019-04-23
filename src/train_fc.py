import os
import argparse
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from skimage.transform import resize


## CLI args
## Currently only 1 train is supported, rest is WIP
'''
parser = argparse.ArgumentParser(description='SNR Lego bricks classification 2019.')
parser.add_argument('task', type=int, help='1|2|3|4', default='1')

args = parser.parse_args()
task = args.task

'''

task = 1

## The actual learning

DATA_PATH = './data/'
IMG_PATH = os.path.join(DATA_PATH, 'Cropped Images')
BATCH_SIZE = 32
TARGET_SIZE = 192


def resize_pad(img):
    old_size = img.shape[:2]
    ratio = float(TARGET_SIZE) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = resize(img, output_shape=new_size, mode='edge', preserve_range=True)

    delta_w = TARGET_SIZE - new_size[1]
    delta_h = TARGET_SIZE - new_size[0]
    padding = (
    (delta_h // 2, delta_h - (delta_h // 2)), (delta_w // 2, delta_w - (delta_w // 2)), (0, 0))

    img = np.pad(img, padding, 'edge')

    return img


def preprocessing_train(x):
    x = resize_pad(x)
    return x


def preprocessing_val(x):
    x = resize_pad(x)
    return x


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_train,
    # rescale=1. / 255,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    validation_split=0.1
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_val,
    # rescale=1. / 255,
    validation_split=0.1
)

train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'), dtype=str)
test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'), dtype=str)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_PATH,
    x_col='filename',
    y_col='brick_type',
    target_size=(TARGET_SIZE, TARGET_SIZE),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='training',
    seed=0
)

validation_generator = test_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_PATH,
    x_col='filename',
    y_col='brick_type',
    target_size=(TARGET_SIZE, TARGET_SIZE),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='validation',
    seed=0
)

labels = (train_generator.class_indices)

# build model
inputs = Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
base_model = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=inputs,
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
    pooling='avg')

# freeze all layers of base model
for layer in base_model.layers:
    layer.trainable = False  # trainable has to be false in order to freeze the layers

predictions = None

if task == 1:
    x = Dense(256, activation='relu')(base_model.output)
    x = Dropout(.25)(x)
    predictions = Dense(len(labels), activation='softmax')(x)

if task == 2:
    base_model.layers[-1].trainable = True
    x = Dense(256, activation='relu')(base_model.output)
    x = Dropout(.25)(x)
    predictions = Dense(len(labels), activation='softmax')(x)

if task == 3:
    for layer in base_model.layers:
        layer.trainable = True
    x = Dense(256, activation='relu')(base_model.output)
    x = Dropout(.25)(x)
    predictions = Dense(len(labels), activation='softmax')(x)


model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()

# train model
model.fit_generator(train_generator, epochs=10,
                    validation_data=validation_generator,
                    steps_per_epoch=len(train_generator),
                    validation_steps=len(validation_generator))

# save model
model.save('model_fc_' + str(mode) + '.h5')
