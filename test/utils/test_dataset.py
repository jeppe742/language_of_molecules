from utils.dataloader import QM9Dataset, DataLoader
import numpy as np
from utils.dummy_data import DummyDataset

def test_dataset():
    qm9 = QM9Dataset('data/adjacency_matrix_train.pkl')

    np.random.seed(0)
    sample = qm9[0]

    assert sample.length == 15
    assert sample.targets == np.array(['C'])
    assert np.array_equal(sample.target_mask, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

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