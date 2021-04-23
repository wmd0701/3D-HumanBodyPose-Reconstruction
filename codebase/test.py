import argparse
import shutil
from collections import defaultdict
import os
from os.path import join, dirname, abspath, exists, basename

import pandas as pd
import torch
from tqdm import tqdm

from checkpoints import CheckpointIO
import config


def make_archive(source, destination):
    if exists(destination):
        os.remove(destination)

    base_name = '.'.join(destination.split('.')[:-1])
    root_dir = dirname(source)
    base_dir = basename(source.strip(os.path.sep))
    fname = shutil.make_archive(base_name, 'zip', root_dir, base_dir)
    assert fname.endswith('.zip'), 'Please install zlib module if not available.'

    return fname


def save_submission_file(results, out_dir, cfg_path):
    submission_dir = join(out_dir, 'submission')
    config.cond_mkdir(submission_dir)

    # save results and configuration
    shutil.copy(cfg_path, join(submission_dir, 'config.yaml'))

    # copy code
    codebase_path = dirname(abspath(__file__))
    if exists(join(submission_dir, 'codebase')):
        shutil.rmtree(join(submission_dir, 'codebase'))
    shutil.copytree(codebase_path, join(submission_dir, 'codebase'))

    # zip files
    submission_dir = abspath(submission_dir)
    zip_file = make_archive(submission_dir, f'{submission_dir}.zip')

    # save results
    results.to_csv(join(out_dir, 'results.csv'),
                   columns=['id'] + key_list,
                   index=False)

    print(f"Your submission is saved in: {join(out_dir, 'results.csv')}")
    print(f'Your code is saved in: {zip_file}')


@torch.no_grad()
def test(cfg, cfg_path, model_file):
    # shortened
    out_dir = cfg['out_dir']
    model_file = model_file if model_file is not None else 'model_best.pt'

    # init variables
    model = config.get_model(cfg)
    trainer = config.get_trainer(model, out_dir, cfg)

    # init datasets
    test_data_loader = config.get_data_loader(cfg, mode='test')

    # load pretrained modes if any
    load_dict = CheckpointIO(out_dir, model=model).safe_load(model_file)
    metric_val_best = load_dict.get('loss_val_best', float('inf'))

    print(f'Current best validation metric: {metric_val_best:.8f}')
    config.cond_mkdir(out_dir)

    # test loop
    results = defaultdict(list)
    file_ids = []
    for batch in tqdm(test_data_loader):
        predictions = trainer.test_step(batch, key_list)
        for key in key_list:
            results[key].append(predictions[key])
        file_ids.extend(batch['file_id'])

    results = {key: torch.cat(results[key], dim=0).cpu().numpy().tolist() for key in key_list}
    results['id'] = file_ids
    results = pd.DataFrame(results)

    save_submission_file(results, out_dir, cfg_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test pipeline.')
    parser.add_argument('config', type=str, help='Path to a config file.')
    parser.add_argument('--model_file', type=str, default=None, help='Overwrite the model path.')
    args = parser.parse_args()

    key_list = ['root_loc', 'root_orient', 'betas', 'pose_body', 'pose_hand']
    test(config.load_config(args), args.config, args.model_file)
