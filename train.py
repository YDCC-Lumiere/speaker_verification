import argparse
import yaml

from torch.utils.data.dataloader import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from model.titanet import TitaNet
from dataset.dataset import SpeakerTrainingDataset, SpeakerTestingDataset, collect_testing_batch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--epochs', type=int, required=True, help='num of epoch')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset file')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of workers')
    parser.add_argument('--valid_dir', type=str, required=True, help='Valid dir')

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = TitaNet(**config['model'], **config['loss'], optimizer_config=config['optimizer'],
                    scheduler_config=config['scheduler'] if 'scheduler' in config else None)

    train_dataset = SpeakerTrainingDataset(args.dataset)
    trail_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    val_dataset = SpeakerTestingDataset(args.valid_dir, args.valid_dataset)
    val_dl = DataLoader(val_dataset, batch_size=32, num_workers=args.num_workers, shuffle=False,
                        collate_fn=collect_testing_batch)

    logger = TensorBoardLogger("exp", name="model_L_SGD_0.001_OneCycleLR_0.08_epochs_250")
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{eer:.3f}-{mindcf:.3f}', save_top_k=-1,
                                          save_weights_only=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(max_epochs=args.epochs, logger=logger,
                      check_val_every_n_epoch=1, devices=[0,1,2,3,4,5,6,7],
                      callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, train_dataloaders=trail_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    main()
