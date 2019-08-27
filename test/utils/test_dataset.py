from utils.dataloader import QM9Dataset, DataLoader
import numpy as np


def test_dataset():
    qm9 = QM9Dataset('data/adjacency_matrix_train.pkl')

    np.random.seed(0)
    sample = qm9[0]

    assert sample.length == 15
    assert sample.targets == np.array(['C'])
    assert np.array_equal(sample.target_mask, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    dl = DataLoader(qm9, batch_size=10)
    a = next(iter(dl))
