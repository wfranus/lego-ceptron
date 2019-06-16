import matplotlib.pyplot as plt

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Nadam

from src.utils import create_data_generator, top_5_accuracy


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

    base_model = None

    if args.task == 1:
        print('Train only last FC layers...')
        base_model = crate_base_model()
        # freeze all layers of base model
        for layer in base_model.layers:
            layer.trainable = False

    if args.task == 2:
        print('Train last conv layer of MobileNetV2 and FC layers...')
        base_model = crate_base_model()
        # freeze all layers of base model except last conv layer;
        # last 4 layers in base model are: (conv2d, batch_norm, relu, avg_pool)
        for layer in base_model.layers[:-4]:
            layer.trainable = False

    if args.task == 3:
        print('Train entire network...')
        # train the whole network
        base_model = crate_base_model(False)

    if args.task == 4:
        raise NotImplementedError

    # always train last fully-connected layers (classifier)
    x = Dropout(.2)(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(.5)(x)
    num_classes = len(train_generator.class_indices)
    predictions = Dense(num_classes, activation='softmax')(x)

    # compile model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Nadam(),
                  metrics=['acc', top_5_accuracy])
    # model.summary()

    # train model
    history = model.fit_generator(train_generator,
                                  epochs=args.epochs,
                                  validation_data=valid_generator)

    # save model
    model.save(f'model_fc_{args.task}.h5')

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
