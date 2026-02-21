import json
import argparse
import os
from trainer.BaseTrainer import train


def main():
    args_cmd = setup_parser().parse_args()

    base_param = load_json(args_cmd.base_configs)
    model_param = load_json(args_cmd.model_configs)

    args = {**base_param, **model_param}

    if os.path.exists('/kaggle/input'):
        args['data_root'] = '/kaggle/input/cub200/CUB_200_2011'
    else:
        args['data_root'] = './data/CUB_200_2011'

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Choose your configs.')
    parser.add_argument('--base_configs', type=str, required=True,
                        help='Json file of base settings.')
    parser.add_argument('--model_configs', type=str, required=True,
                        help='Json file of model settings.')
    return parser


if __name__ == '__main__':
    main()