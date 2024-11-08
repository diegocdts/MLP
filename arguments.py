import argparse


def args():
    parser = argparse.ArgumentParser(description='Config files name')

    parser.add_argument('--model',
                        type=str,
                        default='mlp_2a',
                        help='model to be used')

    parser.add_argument('--data',
                        type=str,
                        default='samples_1500',
                        help='name of the dataset to be used')

    parser.add_argument('--train',
                        type=str,
                        default='default',
                        help='train set to be used')

    parsed = parser.parse_args()

    return parsed