import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import numpy as np
import argparse

from factnn.generator.pytorch.datasets import ClusterDataset, DiffuseDataset, EventDataset
from factnn.models.pytorch_models import PointNet2Regressor
from factnn.utils.plotting import plot_energy_confusion, plot_disp_confusion

from trains import Task

task = Task.init(project_name="IACT Regression", task_name="pytorch pointnet++")
task.name += " {}".format(task.id)

logger = task.get_logger()


def calc_confsion_matrix(prediction, truth, log_xy = True):
    if log_xy is True:
        truth = np.log10(truth)
        prediction = np.log10(prediction)

    min_label = np.min(truth)
    min_pred = np.min(prediction)
    max_pred = np.max(prediction)
    max_label = np.max(truth)

    if min_label < min_pred:
        min_ax = min_label
    else:
        min_ax = min_pred

    if max_label > max_pred:
        max_ax = max_label
    else:
        max_ax = max_pred

    limits = [
        min_ax,
        max_ax
    ]

    counts, x_edges, y_edges = np.histogram2d(
        truth,
        prediction,
        bins=[100, 100],
        range=[limits, limits]
    )
    return counts


def test(args, model, device, test_loader, epoch):

    save_test_loss = []
    save_predictions = []
    save_truth = []

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            # sum up batch loss
            if args.loss == "mse":
                test_loss += F.mse_loss(output, data.y, reduction='sum').item()
            else:
                test_loss += F.l1_loss(output, data.y, reduction='sum').item()

            save_test_loss.append(test_loss)
            save_predictions += output
            save_truth += data.y

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f})\n'.format(
        test_loss))
    logger.report_scalar(title='Test Loss {} - epoch'.format(epoch),
                         series='Test Loss', value=test_loss, iteration=epoch)
    confusion_matrix = calc_confsion_matrix(save_predictions, save_truth)
    if args.task == "energy":
        logger.report_confusion_matrix(title="Confusion Matrix", series="Test Output", matrix=confusion_matrix,
                                       iteration=epoch, xaxis=r'$\log_{10}(E_{\mathrm{MC}} \,\, / \,\, \mathrm{GeV})$',
                                       yaxis=r'$\log_{10}(E_{\mathrm{Est}} \,\, / \,\, \mathrm{GeV})$')
    elif args.task == "disp":
        logger.report_confusion_matrix(title="Confusion Matrix", series="Test Output", matrix=confusion_matrix,
                                       iteration=epoch, xaxis=r'$log_{10}(Disp_{MC}) (mm)$',
                                       yaxis=r'$log_{10}(Disp_{Est}) (mm)$')
    elif args.task == "theta":
        logger.report_confusion_matrix(title="Confusion Matrix", series="Test Output", matrix=confusion_matrix,
                                       iteration=epoch, xaxis=r'$log_{10}(Theta_{MC}) (mm)$',
                                       yaxis=r'$log_{10}(Theta_{Est}) (mm)$')
    elif args.task == "phi":
        logger.report_confusion_matrix(title="Confusion Matrix", series="Test Output", matrix=confusion_matrix,
                                       iteration=epoch, xaxis=r'$log_{10}(Phi_{MC}) (mm)$',
                                       yaxis=r'$log_{10}(Phi_{Est}) (mm)$')




def train(args, model, device, train_loader, optimizer, epoch):
    save_loss = []

    model.train()
    for batch_idx, data in enumerate(train_loader):
        data, target = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        if args.loss == "mse":
            loss = F.mse_loss(output, data.y)
        else:
            loss = F.l1_loss(output, data.y)
        loss.backward()

        save_loss.append(loss)

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # Add manual scalar reporting for loss metrics
            logger.report_scalar(title='Train Loss {} - epoch'.format(epoch),
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
    parser.add_argument("--loss", type=str, default="mse", help="loss type, currently mse or mae, default mse")
    parser.add_argument("--task", type=str, default="energy", help="regression task, one of 'energy', 'disp', 'phi', 'theta' for this model")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--log-interval", type=int, default=50, help="number of minibatches between logging")

    return parser


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    num_classes = 1
    transforms = []
    if args.max_points > 0:
        transforms.append(T.FixedPoints(args.max_points))
    if args.augment and args.task == "energy": # Currently haven't figured out rotating the Phi, Theta, and Disp values if points are moved
        transforms.append(T.RandomRotate((-180,180), axis=2)) # Rotate around z axis
        transforms.append(T.RandomFlip(0)) # Flp about x axis
        transforms.append(T.RandomFlip(1)) # Flip about y axis
        transforms.append(T.RandomTranslate(0.001)) # Random jitter
    if args.norm:
        transforms.append(T.NormalizeScale())
    transform = T.Compose(transforms=transforms) if transforms else None
    if args.task == 'disp':
        train_dataset = DiffuseDataset(args.dataset, 'trainval',
                                     pre_transform=None,
                                     transform=transform)
        test_dataset = DiffuseDataset(args.dataset, 'test', pre_transform=None,
                                    transform=transform)
    else:
        train_dataset = EventDataset(args.dataset, 'trainval', include_proton=True, task=args.task, pre_transform=None,
                                     transform=transform)
        test_dataset = EventDataset(args.dataset, 'test', include_proton=True, task=args.task, pre_transform=None,
                                    transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                             num_workers=6)

    config = {"sample_ratio_one": 0.5, "sample_radius_one": 0.2, "sample_max_neighbor": 64, "sample_ratio_two": 0.25,
              "sample_radius_two": 0.4, "fc_1": 1024, "fc_1_out": 512, "fc_2_out": 256, "dropout": 0.5 }
    config = task.connect_configuration(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2Regressor(num_classes, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)
