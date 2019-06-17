import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import keras_modules_injection
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.optimizers import Nadam, SGD
from thundersvm import SVC

from src.classes import CLASSES
from src.utils import create_data_generator, top_5_accuracy, SignalStopping
from src.utils import get_labels
from src.custom_mobilenet_v2 import MobileNetV2 as CustomMobileNetV2


@keras_modules_injection
def create_custom_model(*args, **kwargs):
    return CustomMobileNetV2(*args, **kwargs)


def load_prev_model(file):
    return load_model(file,
                      custom_objects={'top_5_accuracy': top_5_accuracy})


def train(args):
    # prepare data loaders
    train_generator = create_data_generator(split='train',
                                            target_size=args.target_size,
                                            batch_size=args.batch_size,
                                            shuffle=True)
    valid_generator = create_data_generator(split='valid',
                                            target_size=args.target_size,
                                            batch_size=args.batch_size,
                                            shuffle=True)

    # MobileNetV2 requires that input have 3 channels
    inputs = Input(shape=(args.target_size, args.target_size, 3))

    # as base model use MobileNetV2 pre-trained on ImageNet
    def crate_base_model(use_pretrained=True):
        return MobileNetV2(
            include_top=False,
            weights=('imagenet' if use_pretrained else None),
            input_tensor=inputs,
            input_shape=(args.target_size, args.target_size, 3),
            pooling='avg')

    model = None
    loss = 'categorical_crossentropy'
    metrics = ['acc', top_5_accuracy]

    if args.task == '1':
        # In task 1 we use MobileNetV2 pre-trained on ImageNet dataset
        # as a base model, but without top classifier. Then we freeze all layers
        # of the base model and add our custom classifier net.

        print('Train only last FC layers...')
        base_model = crate_base_model()

        # freeze all layers of base model
        for layer in base_model.layers:
            layer.trainable = False

        # create classifier on top of the base model
        x = Dropout(.2)(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(.5)(x)
        num_classes = len(train_generator.class_indices)
        predictions = Dense(num_classes, activation='softmax')(x)

        # compile the model
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss=loss, metrics=metrics, optimizer=Nadam())

    elif args.task == '2':
        # In task 2 we reuse the model we created and trained in task 1.
        # We freeze all layers of the model until the last conv layer. That way
        # we fine-tune the last convolutional block of the MobileNetV2 alongside
        # the top-level classifier. By reusing model from task 1, all layers
        # start with properly trained weights, not random ones, and hence
        # the learned weights in convolutional block should not be wrecked by
        # large gradient updates.
        #
        # This was suggested by Francois Chollet in his blogpost:
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

        print('Train last conv layer of MobileNetV2 and FC layers...')
        model = load_prev_model('model_fc_1.h5')

        # Freeze all layers of the model except last conv block and top classifier.
        # Last 8 layers in the model that will be trained are:
        # (conv2d, batch_norm, relu, avg_pool, dropout, dense, dropout, dense)
        for layer in model.layers[-8:]:
            layer.trainable = True

        # Recompile the model using SGD optimizer instead of adaptive learning
        # rate optimizer to ensure that training is done with a very slow
        # learning rate and previously learned features are not wrecked.
        # We use learning rate 10 times smaller than used previously.
        # Recompiling the model is also necessary for the changes to take effect
        # (some layers are now trainable).
        model.compile(loss=loss, metrics=metrics,
                      optimizer=SGD(lr=2e-4, momentum=0.9, nesterov=True))

    elif args.task == '3a':
        # In task 3a we retrain whole model. The dataset is small and the model
        # is deep so proper learning is difficult when weights of the model
        # are initialized randomly. Because of that we use pre-trained weights
        # from task 2 and mark every layer as trainable.

        print('Train the entire model...')
        model = load_prev_model('model_fc_2.h5')

        for layer in model.layers:
            layer.trainable = True

        model.save('model_fc_3a.h5')

    elif args.task == '3b':
        # In task 3b we use pre-trained model from task 3a and remove the last
        # convolutional block. As a result, all remaining layers, except the
        # top classifier are initialized with pre-trained weights and the top
        # classifier is initialized with random weights.

        print('Train the entire model, but without last conv block...')

        if not os.path.isfile('model_fc_3b_initial.h5'):
            prev_model = load_prev_model('model_fc_3a.h5')

            # create the same model as in task 3a, but without last conv block
            model = create_custom_model(
                include_top=True,
                input_tensor=inputs,
                input_shape=(args.target_size, args.target_size, 3),
                weights=None,
                classes=20)

            # load weights for all layers before the removed conv block
            for i in range(0, len(model.layers) - 5):
                extracted_weights = prev_model.layers[i].get_weights()
                model.layers[i].set_weights(extracted_weights)
                print('loading weights for:', model.layers[i].name)

            for layer in model.layers:
                layer.trainable = True

            model.save('model_fc_3b_initial.h5')

        model = load_prev_model('model_fc_3b_initial.h5')
        model.compile(loss=loss, metrics=metrics,
                      optimizer=SGD(lr=2e-4, momentum=0.9, nesterov=True))

    elif args.task.startswith('4'):
        # In task 4 we use features produced by the last conv block of the model
        # obtained in task 3a and classify these features using SVM for 3 kernel
        # types (linear, quadratic and exponential). We also check how different
        # levels of allowed errors affect the classification results.
        # and retrain

        train_features = None
        valid_features = None

        # extract features using model from task 3a, if not already extracted
        if not os.path.isfile('train_features_3b.npy')\
            or not os.path.isfile('valid_features_3b.npy'):
            base_model = load_prev_model('model_fc_3a.h5')
            model = Model(inputs=base_model.input,
                          outputs=base_model.get_layer('global_average_pooling2d_1').output)
            # do not shuffle training samples before feeding them to the model,
            # so that the order of features is the same as initial order of samples
            train_generator.shuffle = False
            train_features = model.predict_generator(generator=train_generator)
            print('Extracted train features shape:', train_features.shape)
            np.save('train_features_3b.npy', train_features)

            # extract features for samples from validation set
            valid_generator.shuffle = False
            valid_features = model.predict_generator(generator=valid_generator)
            print('Extracted valid features shape:', valid_features.shape)
            np.save('valid_features_3b.npy', valid_features)

        if train_features is None:
            train_features = np.load('train_features_3b.npy')

        if valid_features is None:
            valid_features = np.load('valid_features_3b.npy')

        train_labels = get_labels(split='train')
        train_y = [CLASSES.index(l) for l in train_labels]
        valid_labels = get_labels(split='valid')
        valid_y = [CLASSES.index(l) for l in valid_labels]
        print('Train labels shape:', train_labels.shape)
        print('Valid labels shape:', valid_labels.shape)

        if args.task == '4a':
            svc = SVC(kernel='linear', decision_function_shape='ovo',
                      gpu_id=args.gpu_id, verbose=True, random_state=42)
        elif args.task == '4b':
            svc = SVC(kernel='polynomial', degree=2, decision_function_shape='ovo',
                      gpu_id=args.gpu_id, verbose=True, random_state=42)
        else:
            svc = SVC(kernel='sigmoid', decision_function_shape='ovo',
                      gpu_id=args.gpu_id, verbose=True, random_state=42)

        svc.fit(train_features, train_y)
        svc.save_to_file(f'svc_{args.task}')

        train_acc = svc.score(train_features, train_y)
        valid_acc = svc.score(valid_features, valid_y)
        print(f'Top-1 acc train: {train_acc:.3} / Top-1 acc valid: {valid_acc:.3}')

        # TODO use predict_proba and compute top-5 accuracies

        return

    model.summary()

    # train model
    print('Training started. Press ctrl + \ to safely stop after current epoch.')
    history = model.fit_generator(train_generator,
                                  epochs=args.epochs,
                                  validation_data=valid_generator,
                                  callbacks=[SignalStopping()])

    # save model
    model.save(f'model_fc_{args.task}.h5')
    pickle.dump(history.history, open(f'history_{args.task}.pkl', 'wb'))

    # plot history of training
    if args.plot:
        # Plot accuracy
        plt.figure(0)
        plt.plot(history.history['acc'])
        plt.plot(history.history['top_5_accuracy'])
        plt.plot(history.history['val_acc'])
        plt.plot(history.history['val_top_5_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train top-1', 'Train top-5', 'Valid top-1', 'Valid top-5'],
                   loc='upper left')
        # plt.show()
        plt.savefig(f'history_acc_{args.task}.png')

        # Plot loss
        plt.figure(1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        # plt.show()
        plt.savefig(f'history_loss_{args.task}.png')
