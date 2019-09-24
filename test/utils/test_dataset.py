from utils.dataloader import QM9Dataset, DataLoader
import numpy as np
from utils.dummy_data import DummyDataset

def test_dataset():
    qm9 = QM9Dataset('data/adjacency_matrix_train.pkl')

    np.random.seed(0)
    sample = qm9[0]

    assert sample.length == 19
    assert sample.targets == np.array(['N'])
    assert np.array_equal(sample.target_mask, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert sample.adj.max()==1

    dl = DataLoader(qm9, batch_size=10)
    a = next(iter(dl))

    qm9 = QM9Dataset('data/adjacency_matrix_train.pkl', bond_order=True)

    np.random.seed(0)
    sample = qm9[0]

    assert sample.length == 19
    assert sample.targets == np.array(['N'])
    assert np.array_equal(sample.target_mask, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert sample.adj.max()==2
    dl = DataLoader(qm9, batch_size=10)
    a = next(iter(dl))



def test_dummy():
    dummy = DummyDataset(num_classes=2, num_samples=1000, max_length=15, ambiguity=True)
    sample = dummy[0]

    
    dummy = DummyDataset(num_classes=4, num_samples=1000, max_length=15, num_bondtypes=2)
    sample = dummy[0]


    dummy = DummyDataset(num_classes=3, num_samples=1000, max_length=15, ambiguity=True)
    sample = dummy[0]
    a=1