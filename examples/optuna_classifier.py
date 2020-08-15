import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, DenseDataLoader
import numpy as np
import argparse

from factnn.generator.pytorch.datasets import (
    ClusterDataset,
    DiffuseDataset,
    EventDataset,
)
from factnn.models.pytorch_models import PointNet2, PointNet2Segmenter

import optuna

"""
from trains import Task

task = Task.init(project_name="IACT Classification", task_name="pytorch pointnet++", output_uri="/mnt/T7/")
task.name += " {}".format(task.id)


logger = task.get_logger()
"""


def test(args, model, device, test_loader):
    save_test_loss = []
    save_correct = []

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, data.y, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(data.y.view_as(pred)).sum().item()

            save_test_loss.append(test_loss)
            save_correct.append(correct)

    test_loss /= len(test_loader)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader), 100.0 * correct / len(test_loader)
        )
    )


"""
    logger.report_histogram(
        title="Test Histogram",
        series="correct",
        iteration=1,
        values=save_correct,
        xaxis="Test",
        yaxis="Correct",
    )

    # Manually report test loss and correct as a confusion matrix
    matrix = np.array([save_test_loss, save_correct])
    logger.report_confusion_matrix(
        title="Confusion matrix",
        series="Test loss / correct",
        matrix=matrix,
        iteration=1,
    )
"""


def train(args, model, device, train_loader, optimizer, epoch):
    save_loss = []
    total_loss = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=-1), data.y)
        loss.backward()

        # save_loss.append(loss.item())

        optimizer.step()
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {}\tLoss: {:.6f} \t Average loss {:.6f}".format(
                    epoch, loss.item(), total_loss / (batch_idx + 1)
                )
            )


"""           # Add manual scalar reporting for loss metrics
            logger.report_scalar(
                title="Training Loss {} - epoch".format(epoch),
                series="Loss",
                value=loss.item(),
                iteration=batch_idx,
            )
"""


def default_argument_parser():
    """
    Create a parser with some common arguments.

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
        default="no_clean",
        help="cleanliness value, one of 'no_clean', "
             "'clump5','clump10', 'clump15', 'clump20', "
             "'core5', 'core10', 'core15', 'core20'",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--seed", type=int, default=1337, help="random seed for numpy")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="number of minibatches between logging",
    )

    return parser


def objective(trial):

    # Generate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "sample_ratio_one": trial.suggest_uniform("sample_ratio_one", 0.1, 0.9),
        "sample_radius_one": trial.suggest_uniform("sample_radius_one", 0.1, 0.9),
        "sample_max_neighbor": trial.suggest_int("sample_max_neighbor", 8, 96),
        "sample_ratio_two": trial.suggest_uniform("sample_ratio_two", 0.1, 0.9),
        "sample_radius_two": trial.suggest_uniform("sample_radius_two", 0.1, 0.9),
        "fc_1": trial.suggest_int("fc_1", 256, 1024),
        "fc_1_out": trial.suggest_int("fc_1_out", 64, 512),
        "fc_2_out": trial.suggest_int("fc_2_out", 32, 256),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.9),
    }

    model = PointNet2(2, config).to(device)

    # generate optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    model.train()
    for epoch in range(10):
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(F.log_softmax(output, dim=-1), data.y)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for b, data in enumerate(test_loader):
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(data.y.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    np.random.seed(args.seed)
    transforms = []
    if args.max_points > 0:
        transforms.append(T.FixedPoints(args.max_points))
    if args.augment:
        transforms.append(T.RandomRotate((-180, 180), axis=2))  # Rotate around z axis
        transforms.append(T.RandomFlip(0))  # Flp about x axis
        transforms.append(T.RandomFlip(1))  # Flip about y axis
        transforms.append(T.RandomTranslate(0.0001))  # Random jitter
    if args.norm:
        transforms.append(T.NormalizeScale())
    transform = T.Compose(transforms=transforms) if transforms else None
    train_dataset = EventDataset(
        args.dataset,
        "train",
        include_proton=True,
        task="separation",
        cleanliness=args.clean,
        pre_transform=None,
        transform=transform,
        balanced_classes=True,
        fraction=0.3,
    )
    print(len(train_dataset))
    test_dataset = EventDataset(
        args.dataset,
        "val",
        include_proton=True,
        task="separation",
        cleanliness=args.clean,
        pre_transform=None,
        balanced_classes=True,
        transform=transform,
        fraction=0.5,
    )
    print(len(test_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=12,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=12,
    )
    study = optuna.create_study(study_name="pointnet_classifier_small", direction="maximize", storage="sqlite:///pointnetClassifier.db", load_if_exists=True, pruner=optuna.pruners.HyperbandPruner(max_resource="auto"))
    study.optimize(objective, n_trials=500)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig = optuna.visualization.plot_param_importances(study=study)
    fig.show()