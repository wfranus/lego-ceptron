from argparse import ArgumentParser

from src.train import train
from src.predict import predict

if __name__ == '__main__':
    # CLI args
    parser = ArgumentParser(description='SNR Lego bricks classification 2019.')
    parser.add_argument('mode', choices=('train', 'predict'), help='Run mode.')
    parser.add_argument('task', choices=('1', '2', '3a', '3b', '4a', '4b', '4c'),
                        default='1',
                        help='Task number: 1 - train FC layers only, '
                        '2 - train last conv layer + FC, 3a - train whole model, '
                        '3b - train whole model with last conv block removed, '
                        '4a - train SVC (linear) on features from last conv, '
                        '4b - train SVC (polynomial) on features from last conv, '
                        '4c - train SVC (rbf) on features from last conv.')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-ts', '--target_size', type=int, default=128,
                        choices=[96, 128, 160, 192, 224],
                        help='Target input image size.')
    parser.add_argument('--plot', action='store_true',
                        help='If set, plot with history of training will be '
                             'saved in history_<task>.png')
    parser.add_argument('--out_dir', help='directory for output files during'
                                          'evaluation')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id to use by thundersvm')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
