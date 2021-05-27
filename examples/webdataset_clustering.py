import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u
import numpy as np
import argparse

from factnn.generator.pytorch.datasets import (
    ClusterDataset,
    DiffuseDataset,
    EventDataset,
)
from factnn.models.pytorch_models import PointNet2Segmenter
#from torch_points3d.applications.pointnet2 import PointNet2
import webdataset as wds

def test(model, device, test_loader):
    save_test_loss = []
    save_correct = []

    model.eval()
    y_mask = test_loader.dataset.y_mask
    ious = [[] for _ in range(len(test_loader.dataset.categories))]
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1)

            i, u = i_and_u(pred, data.y, test_loader.dataset.num_classes, data.batch)
            iou = i.cpu().to(torch.float) / u.cpu().to(torch.float)
            iou[torch.isnan(iou)] = 1

            # Find and filter the relevant classes for each category.
            for iou, category in zip(iou.unbind(), data.category.unbind()):
                ious[category.item()].append(iou[y_mask[category]])

        # Compute mean IoU.
        ious = [torch.stack(iou).mean(0).mean(0) for iou in ious]
        save_test_loss.append(torch.tensor(ious).mean().item())
        save_correct.append(ious)

        # Add manual scalar reporting for loss metrics
        # for i, iou in enumerate(ious):
        #    logger.report_scalar(
        #        title="Test Class IoU".format(epoch),
        #        series=f"Class {i} IoU",
        #        value=iou.item(),
        #        iteration=1,
        #    )
        # logger.report_scalar(
        #    title="Test Mean IoU".format(epoch),
        #    series="Mean IoU.",
        #    value=torch.tensor(ious).mean().item(),
        #    iteration=1,
        # )


def train(args, model, device, train_loader, optimizer, epoch):
    save_loss = []

    model.train()
    total_loss = correct_nodes = total_nodes = 0
    for batch_idx, data in enumerate(train_loader):
        data, target = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()

        save_loss.append(loss)

        optimizer.step()
        total_loss += loss.item()
        correct_nodes += output.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
        if batch_idx % args.log_interval == 0:
            print(
                f"[{batch_idx + 1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} "
                f"Train Acc: {correct_nodes / total_nodes:.4f}"
            )
            # Add manual scalar reporting for loss metrics
            # logger.report_scalar(
            #    title="Loss {} - epoch".format(epoch),
            #    series="Loss",
            #    value=loss.item(),
            #    iteration=batch_idx,
            # )
            # logger.report_scalar(
            #    title="Train Accuracy {} - epoch".format(epoch),
            #    series="Train Acc.",
            #    value=correct_nodes / total_nodes,
            #    iteration=batch_idx,
            # )
            total_loss = correct_nodes = total_nodes = 0


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="whether to augment input data, default False",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="whether to normalize point locations, default False",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="max number of sampled points, if > 0, default 0",
    )
    parser.add_argument(
        "--dataset", type=str, default="", help="path to dataset folder"
    )
    parser.add_argument(
        "--clean",
        type=str,
        default="core20",
        help="cleanliness value, used to load valid filenames, should be the option with the least amount of available files, one of 'no_clean', "
             "'clump5','clump10', 'clump15', 'clump20', "
             "'core5', 'core10', 'core15', 'core20'",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="number of minibatches between logging",
    )

    return parser


def decode_to_torch(sample):
    result = dict(__key__=sample["__key__"])
    for key, value in sample.items():
        if key == "points.pyd":
            result["data"] = torch.from_numpy(value)
        elif key == "mask.pyd":
            result["target"] = torch.from_numpy(value)
    return result


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    transforms = []
    if args.max_points > 0:
        transforms.append(T.FixedPoints(args.max_points))
    if args.augment:
        transforms.append(T.RandomRotate((-180, 180), axis=2))  # Rotate around z axis
        transforms.append(T.RandomFlip(0))  # Flp about x axis
        transforms.append(T.RandomFlip(1))  # Flip about y axis
    if args.norm:
        transforms.append(T.NormalizeScale())
    transform = T.Compose(transforms=transforms) if transforms else None

    dataset = wds.WebDataset(args.dataset).shuffle(1000)

    dataset = wds.Processor(dataset, wds.map, decode_to_torch)

    dataloader = DataLoader(dataset.batched(args.batch_size), num_workers=4, batch_size=None)

    config = {
        "sample_ratio_one": 0.5,
        "sample_radius_one": 0.2,
        "sample_max_neighbor": 64,
        "sample_ratio_two": 0.25,
        "sample_radius_two": 0.4,
        "fc_1": 128,
        "fc_2": 64,
        "dropout": 0.5,
        "knn_num": 3,
    }
    #config = task.connect_configuration(config)
    labels = {"Background": 0}
    if args.clump_dataset:
        labels["Clump"] = 1
        labels["Core"] = 2
        num_classes = 3
    else:
        labels["Core"] = 1
        num_classes = 2
    #task.connect_label_enumeration(labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2Segmenter(num_classes, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
