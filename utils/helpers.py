import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
from tqdm import tqdm

ATOMS = ['H','C','O','N','F','M']
atom2int = {atom: (i+1) for i, atom in enumerate(ATOMS)}
int2atom = {(i+1): atom for i, atom in enumerate(ATOMS)}

def download_files():
    import wget
    import tarfile
    from tqdm import tqdm
    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm9.csv"
    print("Downloading ...")
    wget.download(url, "data/qm9.csv")

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or int(length.max().item())
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) >= length.unsqueeze(1)
    mask = mask.float()
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)

    #mask = (mask - 1)
    return mask


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def cross_entropy(q, y):
    '''
    Arg:
        q, array: probability distribution of model
        y, int: Label of an observation
    '''
    # Convert numpy array to int
    if isinstance(y, np.ndarray):
        if y.dtype != np.int:
            y = y.astype(np.int)

    N = len(y)
    return -sum([np.log(q_n[y_n]) for q_n, y_n in zip(q, y)])/N


def run_cross_validation(text, labels, model):
    cv = StratifiedKFold(n_splits=4, shuffle=True)

    metrics = {
        'f1_micro': [],
        'f1_macro': [],
        'cross-entropy': [],
        'perplexity': []
    }

    for train_idx, val_idx in cv.split(text, labels):

        model.fit(text[train_idx], labels[train_idx])

        # Create a list of predictions
        predictions, probabilities = model.predict(text[val_idx])

        # calculate F1 score
        metrics['f1_micro'].append(f1_score(labels[val_idx], predictions, average='micro'))
        metrics['f1_macro'].append(f1_score(labels[val_idx], predictions, average='macro'))

        # Calculate cross-entropy and perplexity
        metrics['cross-entropy'].append(cross_entropy(probabilities, labels[val_idx]))
        metrics['perplexity'].append(np.exp(cross_entropy(probabilities, labels[val_idx])))

    return metrics


def scaffold_split(dataset, frac_train=0.7, frac_valid=0.15, frac_test=0.15 , random_state=42):
    from rdkit.Chem.Scaffolds import MurckoScaffold

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffolds = {}
    errors=0
    for ind, data in enumerate(dataset):
      
      smiles = data[4]
      
      try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles)
      except:
        errors+=1
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds, valid_inds, test_inds = [], [], []
 

    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set

    train = [dataset[train_idx] for train_idx in train_inds]
    valid = [dataset[valid_idx] for valid_idx in valid_inds]
    test = [dataset[test_idx] for test_idx in test_inds]


    return train, valid, test

def plot_prediction(smiles, atoms, targets, predictions):
  import tempfile
  import rdkit.Chem as chem
  import rdkit.Chem.rdchem as rdchem
  import rdkit.Chem.Draw as draw
  from PIL import Image
  import matplotlib.pyplot as plt

  mol = chem.MolFromSmiles(smiles)

  chem.Kekulize(mol, clearAromaticFlags=True)
  mol = chem.AddHs(mol)
  mol = rdchem.RWMol(mol)

  masked_idx = []
  masked_color = {}
  # for i, (atom, target, prediction) in enumerate(zip(atoms, targets, predictions)):
  #   if atom=='M':
  #     masked_idx += [i]

  #     if target==prediction:
  #       masked_color[i] = (0,1,0)
  #     else:
  #       masked_color[i] = (1,0,0)

  masked_idx = [idx for (idx,atom) in enumerate(atoms) if atom=='M']


  #replace the masks with the predictions
  for i, target,prediction in zip(masked_idx,targets,predictions):
    if target==prediction:
      masked_color[i] = (0,1,0)
    else:
      masked_color[i] = (1,0.3,0)
  
  fig=plt.figure()
  fig.add_subplot(1,2,1)
  plt.title('Input')
  plot_molecule(mol, highlight_atoms=masked_idx, highlight_atom_colors=masked_color,show=False)

    
  #replace the masks with the predictions
  for i, target,prediction in zip(masked_idx,targets,predictions):
    mol.ReplaceAtom(i, rdchem.Atom(int2atom[prediction.item()+1]))
    # if mol.GetBondWithIdx(i).GetIsAromatic():
    #   mol.GetAtomWithIdx(i).SetIsAromatic(1)

  fig.add_subplot(1,2,2)
  plt.title('Prediction')
  plot_molecule(mol,highlight_atoms=masked_idx, highlight_atom_colors=masked_color)


def plot_molecule(molecule, highlight_atoms=[], highlight_atom_colors={}, show=True, inchi=False):
  import tempfile
  import rdkit.Chem as chem
  import rdkit.Chem.rdchem as rdchem
  import rdkit.Chem.Draw as draw
  from PIL import Image
  import matplotlib.image as mpimage
  import matplotlib.pyplot as plt

  if isinstance(molecule,str):
    if inchi:
      molecule = chem.MolFromInchi(molecule)
    else:
      molecule = chem.MolFromSmiles(molecule)
    chem.Kekulize(molecule,clearAromaticFlags=True)
    chem.SanitizeMol(molecule)
    molecule = chem.AddHs(molecule)
  draw.rdDepictor.Compute2DCoords(molecule)

  drawer = draw.rdMolDraw2D.MolDraw2DCairo(500,500)
  tm = draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors)
  drawer.FinishDrawing()

  png = drawer.GetDrawingText()

  with tempfile.NamedTemporaryFile() as tmp:
      tmp.write(png)
      image = mpimage.imread(tmp)
      #image = Image.open(tmp)
      plt.imshow(image, interpolation="bilinear")
      plt.axis('off')
      #image.show()

  if show:
    plt.show()