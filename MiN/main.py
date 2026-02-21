import json
import argparse
import os
import subprocess
from trainer.BaseTrainer import train


def download_cub(root_path):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz"
    tgz_path = os.path.join(root_path, "CUB_200_2011.tgz")
    extract_path = os.path.join(root_path, "CUB_200_2011")

    if not os.path.exists(extract_path):
        print("Downloading CUB-200-2011...")
        subprocess.run(["wget", url, "-O", tgz_path], check=True)

        print("Extracting...")
        subprocess.run(["tar", "-xzf", tgz_path, "-C", root_path], check=True)

        print("Done extracting.")

    return extract_path


def main():
    args_cmd = setup_parser().parse_args()

    base_param = load_json(args_cmd.base_configs)
    model_param = load_json(args_cmd.model_configs)

    args = {**base_param, **model_param}

    if os.path.exists('/kaggle'):
        working_root = "/kaggle/working"
        cub_path = download_cub(working_root)
        args['data_root'] = cub_path
    else:
        args['data_root'] = "./data/CUB_200_2011"

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Choose your configs.')
    parser.add_argument('--base_configs', type=str, required=True)
    parser.add_argument('--model_configs', type=str, required=True)
    return parser


if __name__ == '__main__':
    main()