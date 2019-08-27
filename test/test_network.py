from layers.transformer import TransformerLayer, TransformerModel
from utils.dataloader import QM9Dataset, DataLoader
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy


def test_transformer_forward_cpu():
    qm9 = QM9Dataset('data/adjacency_matrix_train.pkl', epsilon_greedy=0.5)

    np.random.seed(0)
    torch.manual_seed(0)

    dl = DataLoader(qm9, batch_size=2)
    sample = next(iter(dl))
    transformer = TransformerModel()
    out = transformer(sample)

    assert torch.equal(out['prediction'], torch.tensor([0, 4, 0, 4, 0, 0, 0]))

    criterion = CrossEntropyLoss()
    targets = sample.targets_num

    assert torch.equal(targets, torch.tensor([[2, 1, 3, 2, 1, 1], [1, 0, 0, 0, 0, 0]]))
    targets = targets[targets != 0]
    targets -= 1
    assert torch.equal(targets, torch.tensor([1, 0, 2, 1, 0, 0, 0]))
    loss = criterion(out['out'], targets)

    assert torch.equal(loss, cross_entropy(out['out'], targets, reduction='none').mean())


def test_transformer_forward_cuda():
    qm9 = QM9Dataset('data/adjacency_matrix_train.pkl')

    np.random.seed(0)
    torch.manual_seed(0)

    dl = DataLoader(qm9, batch_size=1)
    sample = next(iter(dl))
    transformer = TransformerModel().cuda()
    sample.cuda()
    out = transformer(sample)

    assert torch.equal(out['prediction'], torch.tensor([0]).cuda())
