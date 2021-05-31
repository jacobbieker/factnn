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


def train(args, model, device, train_loader, optimizer, epoch):
    save_loss = []
    global_iteration = 0
    max_iterations = 5000

    model.train()
    total_loss = correct_nodes = total_nodes = 0
    for batch_idx, data in enumerate(train_loader):
        global_iteration += 1
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output[0], data.y)
        loss.backward()

        save_loss.append(loss)

        optimizer.step()
        total_loss += loss.item()
        correct_nodes += output[0].argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
        if batch_idx % args.log_interval == 0:
            print(
                f"[{batch_idx + 1}/{max_iterations}] Loss: {total_loss / 8:.4f} "
                f"Train Acc: {correct_nodes / total_nodes:.4f}"
            )
            total_loss = correct_nodes = total_nodes = 0
        if global_iteration % max_iterations == 0:
            break


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
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="number of minibatches between logging",
    )

    return parser


def torch_loads(data):
    import io
    import torch
    stream = io.BytesIO(data)
    return torch.load(stream)


def decode_to_torch(sample):
    from torch_geometric.data import Data
    result = Data(
        pos=sample["points.pth"], y=sample["mask.pth"]
    )  # Just need x,y,z ignore derived features
    return result


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    dataset = wds.WebDataset("/run/media/bieker/T7/fact-train-5-{0000000..0000008}.tar").shuffle(1000).decode()
    #test_dataset = wds.WebDataset(args.dataset)
    dataset = wds.Processor(dataset, wds.map, decode_to_torch)
    #test_dataset = wds.Processor(test_dataset, wds.map, decode_to_torch)

    train_loader = DataLoader(dataset, num_workers=4, batch_size=8)
    #test_loader = DataLoader(test_dataset, num_workers=4, batch_size=1)

    config = {
        "sample_ratio_one": 0.5,
        "sample_radius_one": 0.2,
        "sample_max_neighbor": 64,
        "sample_ratio_two": 0.25,
        "sample_radius_two": 0.4,
        "fc_1": 128,
        "fc_2": 128,
        "dropout": 0.5,
        "knn_num": 3,
    }

    labels = {"Background": 0}
    labels["Clump"] = 1
    labels["Core"] = 2
    num_classes = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2Segmenter(num_classes, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Made Model")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        train(args, model, device, train_loader, optimizer, epoch)
        #test(model, device, test_loader)
