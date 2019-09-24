from utils.helpers import scaffold_split, plot_prediction
import pickle
from utils.dataloader import DataLoader, QM9Dataset
from utils.dummy_data import DummyDataset
import torch

def test_scaffold_split():
    data = pickle.load(open("data/adjacency_matrix_test.pkl","rb"))

    scaffold_split(data)



def test_visualize_smiles():
    dummy = DummyDataset(num_classes=4, num_samples=1000, max_length=15, ambiguity=True,num_masks=2)
    sample = dummy[0]
    sample.plot()

    dl = DataLoader(dummy, batch_size=1)
    batch = next(iter(dl))

    plot_prediction(batch.smiles[0], batch.atoms[0], torch.randint(5,(2,))+1,torch.randint(5,(2,))+1)

    a=1
