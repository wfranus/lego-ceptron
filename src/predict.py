import numpy as np

from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model, Model

from src.utils import create_data_generator, preprocess_input_custom


def evaluate(args):
    model: Model = load_model(f'model_fc_{args.task}.h5')

    test_generator = create_data_generator(split='test',
                                           target_size=args.target_size,
                                           batch_size=args.batch_size)

    score = model.evaluate_generator(test_generator, verbose=1)

    print(f'Evaluation score:\n\tloss: {score[0]}\n\tacc: {score[1]}')


def predict(args):
    model: Model = load_model(f'model_fc_{args.task}.h5')

    test_generator = create_data_generator(split='test',
                                           target_size=args.target_size,
                                           batch_size=args.batch_size)

    preds = model.predict_generator(test_generator, verbose=1)

    save_file = f'preds_{args.task}.npy'
    np.save(save_file, preds)
    print(f'Predictions saved to: {save_file}')


# for debugging
def predict_one(img_path, task):
    model: Model = load_model(f'model_fc_{task}.h5')

    img = load_img(img_path)
    x = img_to_array(img)
    x = preprocess_input_custom(192)(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)

    pred_class_id = np.argmax(preds[0])
    print(f'predicted class: {pred_class_id}, '
          f'probability: {preds[0][pred_class_id]}')


if __name__ == '__main__':
    predict_one('data/Cropped Images/Plate_1x1_Slope/1_Plate_1x1_Slope_180715175157.jpg', 1)
