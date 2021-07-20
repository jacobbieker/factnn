from torch_geometric.data import DataLoader
import argparse

from factnn.models.pytorch_models import LitPointNet2
from optuna.integration import PyTorchLightningPruningCallback
import webdataset as wds
import torch.utils.data as thd
from torch_geometric.data import Dataset


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="number of minibatches between logging",
    )

    return parser


def decode_to_torch(sample):
    from torch_geometric.data import Data
    points = sample["points.pth"]
    mask = sample["mask.pth"]
    mask = mask[mask > 1.1] # Only take core 10 ones
    points = points[mask]
    is_gamma = sample["class.cls"]
    result = Data(
        pos=points, y=is_gamma
    )  # Just need x,y,z ignore derived features
    return result

import optuna


class SampleEqually(thd.IterableDataset, wds.Shorthands, wds.Composable):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        while True:
            for source in sources:
                try:
                    sample = next(source)
                    yield decode_to_torch(sample)
                except StopIteration:
                    return


def objective(trial: optuna.trial.Trial) -> float:

    dataset = wds.WebDataset("/run/media/jacob/data/FACT_Dataset/fact-gamma-10-{0000..0062}.tar").shuffle(20000).decode()
    dataset_2 = wds.WebDataset("/run/media/jacob/data/FACT_Dataset/fact-proton-10-{0000..0010}.tar").shuffle(20000).decode()
    test_dataset_2 = wds.WebDataset("/run/media/jacob/data/FACT_Dataset/fact-gamma-10-{0063..0072}.tar").decode()
    test_dataset = wds.WebDataset("/run/media/jacob/data/FACT_Dataset/fact-proton-10-{0011..0013}.tar").decode()
    dataset = SampleEqually([dataset, dataset_2])
    test_dataset = SampleEqually([test_dataset_2, test_dataset])

    train_loader = DataLoader(dataset, num_workers=16, batch_size=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=1, pin_memory=True)

    # We optimize the number of layers, hidden units in each layer and dropouts.
    config = {
        "sample_ratio_one": trial.suggest_uniform("sample_ratio_one", 0.1, 0.9),
        "sample_radius_one": trial.suggest_uniform("sample_radius_one", 0.1, 0.9),
        "sample_max_neighbor": trial.suggest_int("sample_max_neighbor", 8, 72),
        "sample_ratio_two": trial.suggest_uniform("sample_ratio_two", 0.1, 0.9),
        "sample_radius_two": trial.suggest_uniform("sample_radius_two", 0.1, 0.9),
        "fc_1": trial.suggest_int("fc_1", 128, 256),
        "fc_1_out": trial.suggest_int("fc_1_out", 32, 128),
        "fc_2_out": trial.suggest_int("fc_2_out", 16, 96),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.9),
    }

    num_classes = 2
    import pytorch_lightning as pl
    model = LitPointNet2(num_classes, lr=0.0001, config=config)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=10000,
        limit_train_batches=10000,
        checkpoint_callback=False,
        auto_lr_find=True,
        max_epochs=20,
        gpus=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val/loss")],
    )
    trainer.logger.log_hyperparams(config)
    trainer.tune(model=model, train_dataloader=train_loader, val_dataloaders=test_loader)
    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=test_loader)

    return trainer.callback_metrics["val/loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
             "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100, gc_after_trial=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))




