import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from factnn.generator.pytorch.datasets import ClusterDataset, DiffuseDataset, EventDataset
from factnn.models.pytorch_models import PointNet2_Classifier, PointNet2PartSegmentNet


def train(model, train_loader):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    num_classes = 2
    # net = PointNet2PartSegmentNet(num_classes)
    path = ""
    uncleaned_path = ""
    transform = T.SamplePoints(1024)
    train_dataset = ClusterDataset(path, split='trainval', uncleaned_root=uncleaned_path, pre_transform=None,
                                 transform=None)
    test_dataset = ClusterDataset(path, split='test', uncleaned_root=uncleaned_path, pre_transform=None,
                                transform=None)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2PartSegmentNet(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
        train(model, train_loader)
        test_acc = test(model, test_loader)
        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))
