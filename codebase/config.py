import os
import yaml
from torch.utils.data import DataLoader

from data.dataset import H36MDataset
from regressor.model import BaseModel
from regressor.training import ConvTrainer

from torch import optim


def load_config(args):
    """ Loads configuration file.

    Returns:
        cfg (dict): configuration file
    """
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def cond_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_data_loader(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']

    if mode == 'train':
        subjects = cfg['data']['train_subjects'].split(',')
        batch_size = cfg['training']['batch_size']
    elif mode == 'val':
        subjects = cfg['data']['val_subjects'].split(',')
        batch_size = cfg['training']['batch_size']
    else:
        subjects = ['S9', 'S11']
        batch_size = 1

    dataset = H36MDataset(
        dataset_folder=cfg['data']['dataset_folder'],
        img_folder=cfg['data']['img_folder'],
        subjects=subjects,
        mode=mode,
        img_size=(512, 512)
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg['training'].get('num_workers', 0),
        shuffle=mode == 'train',
        collate_fn=dataset.collate_fn
    )
    return data_loader


def get_model(cfg):
    model = BaseModel.create_model(cfg)
    return model.to(device=cfg['device'])


def get_optimizer(model, cfg):
    """ Create an optimizer. """
    params_fc    = list(model.backbone.fc.parameters())
    params_root  = list(model.nn_root_orient.parameters())
    params_betas = list(model.nn_betas.parameters())
    params_body  = list(model.nn_pose_body.parameters())
    params_hand  = list(model.nn_pose_hand.parameters())

    params = params_fc + params_root + params_betas + params_body + params_hand

    if cfg['training']['optimizer']['name'] == 'SGD':
        optimizer = optim.SGD(  params,
                                lr=cfg['training']['optimizer'].get('lr', 1e-4),
                                weight_decay=cfg['training']['optimizer'].get('wd', 1e-5)
                             )
    elif cfg['training']['optimizer']['name'] == 'Adam':
        optimizer = optim.Adam( params,
                                lr=cfg['training']['optimizer'].get('lr', 1e-4),
                                weight_decay=cfg['training']['optimizer'].get('wd', 1e-5)
                              )
    else:
        raise Exception('Not supported.')

    return optimizer


def get_trainer(model, vis_dir, cfg, optimizer=None):
    """ Create a trainer instance. """

    if cfg['trainer'] == 'conv':
        trainer = ConvTrainer(model, optimizer, vis_dir, cfg)
    else:
        raise Exception('Not supported.')

    return trainer
