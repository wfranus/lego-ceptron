from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Dropout, Input
from keras.models import Model

from src.utils import create_data_generator


def train(args):
    # prepare data loaders
    train_generator = create_data_generator(split='train',
                                            target_size=args.target_size,
                                            batch_size=args.batch_size)
    valid_generator = create_data_generator(split='valid',
                                            target_size=args.target_size,
                                            batch_size=args.batch_size)

    # MobileNetV2 requires that input have 3 channels
    inputs = Input(shape=(args.target_size, args.target_size, 3))

    # as base model use MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        input_shape=(args.target_size, args.target_size, 3),
        pooling='avg')

    if args.task == 1:
        print('Train only last FC layers...')
        # freeze all layers of base model
        for layer in base_model.layers:
            layer.trainable = False

    if args.task == 2:
        print('Train last conv layer of MobileNetV2 and FC layers...')
        # freeze all layers of base model except last conv layer;
        # last 4 layers in base model are: (conv2d, batch_norm, relu, avg_pool)
        for layer in base_model.layers[:-4]:
            layer.trainable = False

    if args.task == 3:
        print('Train entire network...')
        # train the whole network
        pass

    if args.task == 4:
        raise NotImplementedError

    # always train last fully-connected layers
    x = Dense(256, activation='relu')(base_model.output)
    x = Dropout(.4)(x)
    num_classes = len(train_generator.class_indices)
    predictions = Dense(num_classes, activation='softmax')(x)

    # compile model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    # train model
    model.fit_generator(train_generator,
                        epochs=args.epochs,
                        validation_data=valid_generator)

    # save model
    model.save(f'model_fc_{args.task}.h5')
