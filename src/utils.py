import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.preprocess_pad  # performes monkeypatching of load_img function

from keras.applications import mobilenet_v2
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from src.classes import CLASSES


def create_data_generator(data_dir='./data', split='train',
                          target_size=192, batch_size=32, seed=0, shuffle=True):
    assert split in ['train', 'test', 'valid']

    if split == 'train':
        # augment train set using transformations of images
        generator = ImageDataGenerator(
            preprocessing_function=mobilenet_v2.preprocess_input,
            # zoom_range=[1.0, 2.0],  # zoom-out, never zoom-in
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=90,

            # transformations below require fitting generator on training data first!
            # zca_whitening=True,
            # featurewise_center=True,
            # featurewise_std_normalization=True,
        )
    else:
        generator = ImageDataGenerator(
            preprocessing_function=mobilenet_v2.preprocess_input,
        )

    # load dataframe with paths to imgs and labels
    df = pd.read_csv(os.path.join(data_dir, f'{split}.csv'), dtype=str)

    # create data iterator that loads imgs in batch and aplies transformations
    generator = generator.flow_from_dataframe(
        dataframe=df,
        directory=data_dir,
        x_col='filename',
        y_col='brick_type',
        target_size=(target_size, target_size),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        classes=CLASSES
    )

    return generator


def top_5_accuracy(y_true, y_pred):
    """Calculates top-5 accuracy.
    Uses Keras implementation.
    Suitable to be passed to model.compile function
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def top_k_accuracy_score(y_true, y_pred, k=5, normalize=True):
    """Calculates top-k accuracy.
    Implementation taken from scikit-learn pull request.
    Suitable to be used with predictions returned by model.predict function.

    source: https://github.com/scikit-learn/scikit-learn/blob/4685cb5c50629aba4429f6701585f82fc3eee5f7/sklearn/metrics/classification.py
    """
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    num_obs, num_labels = y_pred.shape
    idx = num_labels - k - 1
    counter = 0
    argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter / num_obs
    else:
        return counter


def plot_confusion_matrix(y_true, y_pred, classes, path=None, title=None,
                          print_to_stdout=False):
    """Plots the confusion matrix and optionally prints it to stdout.

    source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-download-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    if print_to_stdout:
        print('Confusion matrix')
        print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Save
    if path:
        plt.savefig(path)

    return ax
