import argparse
from os.path import join

from tensorboardX import SummaryWriter

from checkpoints import CheckpointIO
import config


def train(cfg, model_file):
    # shortened
    out_dir = cfg['out_dir']
    model_file = model_file if model_file is not None else 'model_best.pt'
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    model_selection_metric = cfg['training'].get('model_selection_metric', 'v2v_l2')

    # init variables
    model = config.get_model(cfg)
    optimizer = config.get_optimizer(model, cfg)
    trainer = config.get_trainer(model, out_dir, cfg, optimizer)
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

    # init datasets
    train_data_loader = config.get_data_loader(cfg, mode='train')
    val_data_loader = config.get_data_loader(cfg, mode='val')

    # load pretrained modes if any
    load_dict = checkpoint_io.safe_load(model_file)
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get('loss_val_best', float('inf'))

    # prepare loggers
    print(model, f'\n\nTotal number of parameters: {sum(p.numel() for p in model.parameters()):d}')
    print(f'Current best validation metric: {metric_val_best:.8f}')

    config.cond_mkdir(out_dir)
    logger = SummaryWriter(join(out_dir, 'logs'))

    # training loop
    while True:
        epoch_it += 1
        for batch in train_data_loader:
            it += 1

            loss_dict = trainer.train_step(batch)
            loss = loss_dict['total_loss']
            for k, v in loss_dict.items():
                logger.add_scalar(f'train/{k}', v, it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                print(f'[Epoch {epoch_it:02d}] it={it:05d}, loss={loss:.8f}')

            # Save checkpoint
            if checkpoint_every > 0 and (it % checkpoint_every) == 0:
                print('Saving checkpoint')
                checkpoint_io.save(f'model_{it:d}.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0 and it > 0:
                eval_dict, val_img = trainer.evaluate(val_data_loader)
                # log eval metric
                metric_val = eval_dict[model_selection_metric]
                print(f'Validation metric ({model_selection_metric}): {metric_val:.8f}')
                # log eval images
                if val_img is not None:
                    logger.add_image(f'val/renderings', val_img, it)

                for k, v in eval_dict.items():
                    logger.add_scalar(f'val/{k}', v, it)

                if metric_val < metric_val_best:
                    metric_val_best = metric_val
                    print(f'New best model (loss {metric_val_best:.8f})')
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pipeline.')
    parser.add_argument('config', type=str, help='Path to a config file.')
    parser.add_argument('--model_file', type=str, default=None, help='Overwrite the model path.')
    _args = parser.parse_args()

    train(config.load_config(_args), _args.model_file)
