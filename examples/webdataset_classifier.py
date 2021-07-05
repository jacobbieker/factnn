from torch_geometric.data import DataLoader
import argparse

from factnn.models.pytorch_models import LitPointNet2
import webdataset as wds


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
    is_gamma = sample["class.cls"]
    result = Data(
        pos=points, y=is_gamma
    )  # Just need x,y,z ignore derived features
    return result


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    dataset = wds.WebDataset("/run/media/jacob/data/FACT_Dataset/fact-train-10-{0000..0040}.tar").shuffle(2000).decode()
    test_dataset = wds.WebDataset("/run/media/jacob/data/FACT_Dataset/fact-test-5-{0000..0017}.tar").decode()
    dataset = wds.Processor(dataset, wds.map, decode_to_torch)
    test_dataset = wds.Processor(test_dataset, wds.map, decode_to_torch)

    train_loader = DataLoader(dataset, num_workers=12, batch_size=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=1, pin_memory=True)

    config = {
        "sample_ratio_one": 0.5,
        "sample_radius_one": 0.2,
        "sample_max_neighbor": 64,
        "sample_ratio_two": 0.25,
        "sample_radius_two": 0.4,
        "fc_1": 512,
        "fc_1_out": 256,
        "fc_2_out": 128,
        "dropout": 0.5,
    }

    num_classes = 2
    import pytorch_lightning as pl
    model = LitPointNet2(num_classes, lr=0.001, config=config)
    trainer = pl.Trainer(gpus=1, precision=32, auto_lr_find=True, accumulate_grad_batches=64)
    trainer.tune(model=model, train_dataloader=train_loader, val_dataloaders=test_loader)
    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=test_loader)
