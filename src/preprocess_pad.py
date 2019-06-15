import numpy as np
import keras_preprocessing.image

from skimage.transform import resize


def load_and_pad_img(path, grayscale=False, color_mode='rgb', target_size=None,
                     interpolation='nearest'):
    """This function will be used instead of the load_img function imported by
    keras_preprocessing.image.iterator.

    Image is loaded in original size, then padded to target size using
    background color. Aspect ratio of the lego brick is kept unchanged.
    """
    # Load original size image using Keras
    img = keras_preprocessing.image.utils.load_img(path,
                                                   grayscale=grayscale,
                                                   color_mode=color_mode,
                                                   target_size=None,
                                                   interpolation=interpolation)

    img = np.array(img)
    old_size = img.shape[:2]
    ratio = float(target_size[0]) / max(old_size)
    new_size = old_size

    # scale down if any dim of the original size is greater than target size
    if ratio < 1.0:
        new_size = tuple([int(x * ratio) for x in old_size])
        img = resize(img, output_shape=new_size, mode='edge', preserve_range=True)

    # pad images to the target size
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    padding = (
        (delta_h // 2, delta_h - (delta_h // 2)),
        (delta_w // 2, delta_w - (delta_w // 2)),
        (0, 0)
    )

    img = np.pad(img, padding, 'edge')

    return img


# Monkey patch
keras_preprocessing.image.iterator.load_img = load_and_pad_img
