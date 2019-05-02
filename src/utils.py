import os

import numpy as np
import pandas as pd
from keras.applications import mobilenet_v2
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize


def preprocess_input_custom(target_size):
    """Resize and pad input image, so that ratio between height and width
    of a lego brick on the image remain unchanged.

    This function should be used instead of **preprocess_input** from
    keras.applications.mobilenet_v2, because it performs additional
    preprocessing and calls **preprocess_input** internally.
    """
    def preprocess_input(img):
        old_size = img.shape[:2]
        ratio = float(target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        img = resize(img, output_shape=new_size, mode='edge', preserve_range=True)

        delta_w = target_size - new_size[1]
        delta_h = target_size - new_size[0]
        padding = (
            (delta_h // 2, delta_h - (delta_h // 2)),
            (delta_w // 2, delta_w - (delta_w // 2)),
            (0, 0)
        )

        img = np.pad(img, padding, 'edge')
        img = mobilenet_v2.preprocess_input(img)

        return img

    return preprocess_input


def create_data_generator(data_dir='./data', split='train',
                          target_size=192, batch_size=32, seed=0, **kwargs):
    assert split in ['train', 'test', 'valid']

    if split == 'train':
        # augment train set using transformations of images
        generator = ImageDataGenerator(
            preprocessing_function=preprocess_input_custom(target_size),
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=90,
            **kwargs
        )
    else:
        generator = ImageDataGenerator(
            preprocessing_function=preprocess_input_custom(target_size),
        )

    df = pd.read_csv(os.path.join(data_dir, f'{split}.csv'), dtype=str)

    generator = generator.flow_from_dataframe(
        dataframe=df,
        directory=data_dir,
        x_col='filename',
        y_col='brick_type',
        target_size=(target_size, target_size),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        seed=seed
    )

    return generator
