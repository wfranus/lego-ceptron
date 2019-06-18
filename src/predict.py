import os
import numpy as np

from keras.models import load_model, Model
from sklearn.metrics import accuracy_score, classification_report, zero_one_loss
from thundersvm import SVC

from src.classes import CLASSES
from src.utils import create_data_generator, get_labels
from src.utils import plot_confusion_matrix, top_5_accuracy, top_k_accuracy_score
from src.preprocess_pad import load_and_pad_img


def predict(args):
    """Loads model from file, makes predictions and computes metrics.

    All created files are saved to args.out_dir directory if provided,
    or to results_<task> otherwise.

    Creates files:
    conf_matrix.png file with confusion matrix,
    report.txt with various metrics,
    preds_{task}.npy with raw predictions.
    """
    if args.task.startswith('4'):
        test_features = np.load('test_features_3b.npy')
        test_labels = get_labels(split='test')
        y_true = np.array([CLASSES.index(l) for l in test_labels])

        out_dir = args.out_dir or f'results_{args.task}'

        for c in [0.001, 0.01, 0.1, 1.0, 10]:
            svc = SVC()
            svc.load_from_file(f'svc_{args.task}_C_{c}')

            y_pred = svc.predict(test_features)

            evaluate(y_true, y_pred, None, CLASSES, os.path.join(out_dir, f'C_{c}'))
    else:
        model: Model = load_model(f'model_fc_{args.task}.h5',
                                  custom_objects={'top_5_accuracy': top_5_accuracy})

        test_generator = create_data_generator(split='test',
                                               target_size=args.target_size,
                                               batch_size=args.batch_size,
                                               shuffle=False)

        # get predictions
        preds = model.predict_generator(test_generator, verbose=1)

        # create output directory
        out_dir = args.out_dir or f'results_{args.task}'
        os.makedirs(out_dir, exist_ok=True)

        # save numpy array with predictions
        save_file = os.path.join(out_dir, f'preds_{args.task}.npy')
        np.save(save_file, preds)
        print(f'Predictions saved to: {save_file}')

        # first, prepare y_pred, y_true and class names
        # y_pred are classes predicted with the highest probability
        y_pred = np.array([np.argmax(x) for x in preds])
        # since we did not shuffle data in data generator,
        # classes attribute of the generator contains true labels for each sample
        y_true = np.array(test_generator.classes)
        # class_names = list(test_generator.class_indices.keys())
        # class_names.sort(key=lambda x: test_generator.class_indices[x])

        evaluate(y_true, y_pred, preds, CLASSES, out_dir)


def evaluate(y_true, y_pred, preds=None, classes=CLASSES, out_dir='results'):
    # create output directory
    os.makedirs(out_dir, exist_ok=True)

    # calculate evaluation metrics
    conf_mat_file = os.path.join(out_dir, 'conf_matrix.png')
    plot_confusion_matrix(y_true, y_pred, np.array(classes), conf_mat_file,
                          print_to_stdout=False)
    print(f'Confusion matrix saved to: {conf_mat_file}')

    # classification report returns precision, recall, f-score,
    # support (the number of occurrences of each class in y_true)
    # and micro/macro averages of these metrics for each class
    report_str = classification_report(y_true, y_pred, target_names=classes)

    # compute more metrics and append them to the report
    top_1_acc = accuracy_score(y_true, y_pred)
    error_rate = zero_one_loss(y_true, y_pred)

    if preds is not None:
        # top-5 accuracy computed only for neural models
        top_5_acc = top_k_accuracy_score(y_true, preds, k=5)

    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write(report_str)
        f.write(f'\ntop-1 accuracy: {top_1_acc:.2}')
        if preds is not None:
            f.write(f'\ntop-5 accuracy: {top_5_acc:.2}')
        f.write(f'\nerror rate: {error_rate:.2}\n')

# for debugging
def predict_one(img_path, task):
    model: Model = load_model(f'model_fc_{task}.h5',
                              custom_objects={'top_5_accuracy': top_5_accuracy})

    x = load_and_pad_img(img_path, target_size=(128, 128))
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)

    pred_class_id = np.argmax(preds[0])
    print(f'predicted class: {pred_class_id}, '
          f'probability: {preds[0][pred_class_id]}')


if __name__ == '__main__':
    predict_one('data/Cropped Images/Brick_2x2_L/1_Brick_2x2_L_180708213653.jpg', 1)
