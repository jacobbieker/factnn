import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import numpy as np
import argparse

from factnn.generator.pytorch.datasets import ClusterDataset, DiffuseDataset, EventDataset
from factnn.models.pytorch_models import PointNet2Classifier, PointNet2Segmenter

from trains import Task

task = Task.init(project_name="IACT Classification", task_name="pytorch pointnet++")
task.name += " {}".format(task.id)



logger = task.get_logger()


def test(args, model, device, test_loader):

    save_test_loss = []
    save_correct = []

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, data.y, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(data.y.view_as(pred)).sum().item()

            save_test_loss.append(test_loss)
            save_correct.append(correct)

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader),
        100. * correct / len(test_loader)))

    logger.report_histogram(title='Histogram example', series='correct',
        iteration=1, values=save_correct, xaxis='Test', yaxis='Correct')

    # Manually report test loss and correct as a confusion matrix
    matrix = np.array([save_test_loss, save_correct])
    logger.report_confusion_matrix(title='Confusion matrix example',
        series='Test loss / correct', matrix=matrix, iteration=1)


def train(args, model, device, train_loader, optimizer, epoch):
    save_loss = []

    model.train()
    for batch_idx, data in enumerate(train_loader):
        data, target = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()

        save_loss.append(loss)

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # Add manual scalar reporting for loss metrics
            logger.report_scalar(title='Scalar example {} - epoch'.format(epoch),
                                 series='Loss', value=loss.item(), iteration=batch_idx)


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
    parser.add_argument("--augment", action="store_true", help="whether to augment input data, default False")
    parser.add_argument("--norm", action="store_true", help="whether to normalize point locations, default False")
    parser.add_argument("--max-points", type=int, default=0, help="max number of sampled points, if > 0, default 0")
    parser.add_argument("--dataset", type=str, default="", help="path to dataset folder")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--log-interval", type=int, default=50, help="number of minibatches between logging")

    return parser


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    num_classes = 2
    transforms = []
    if args.max_points > 0:
        transforms.append(T.FixedPoints(args.max_points))
    if args.augment:
        transforms.append(T.RandomRotate((-180,180), axis=2)) # Rotate around z axis
        transforms.append(T.RandomFlip(0)) # Flp about x axis
        transforms.append(T.RandomFlip(1)) # Flip about y axis
        transforms.append(T.RandomTranslate(0.001)) # Random jitter
    if args.norm:
        transforms.append(T.NormalizeScale())
    transform = T.Compose(transforms=transforms) if transforms else None
    train_dataset = EventDataset(args.dataset, 'trainval', include_proton=True, task="Separation", pre_transform=None,
                                 transform=transform)
    test_dataset = EventDataset(args.dataset, 'test', include_proton=True, task="Separation", pre_transform=None,
                                transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                             num_workers=6)

    config = {"sample_ratio_one": 0.5, "sample_radius_one": 0.2, "sample_max_neighbor": 64, "sample_ratio_two": 0.25,
              "sample_radius_two": 0.4, "fc_1": 1024, "fc_1_out": 512, "fc_2_out": 256, "dropout": 0.5 }
    config = task.connect_configuration(config)
    task.connect_label_enumeration({"Gamma": 0, "Proton": 1})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2Classifier(num_classes, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
