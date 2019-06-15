from argparse import ArgumentParser

from src.train import train
from src.predict import predict

if __name__ == '__main__':
    # CLI args
    parser = ArgumentParser(description='SNR Lego bricks classification 2019.')
    parser.add_argument('mode', choices=('train', 'predict'), help='Run mode.')
    parser.add_argument('task', type=int, choices=(1, 2, 3, 4), default=1,
                        help='Task number: 1 - train FC layers only, '
                        '2 - train last conv layer + FC, 3 - train whole model')
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

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
