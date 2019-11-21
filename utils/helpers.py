import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
from tqdm import tqdm
from scipy.special import softmax
from scipy.optimize import least_squares

ATOMS = ['H','C','O','N','F','P','S','Cl','Br','I','M']
atom2int = {atom: (i+1) for i, atom in enumerate(ATOMS)}
int2atom = {(i+1): atom for i, atom in enumerate(ATOMS)}

def download_files(url):
    import wget
    import tarfile
    from tqdm import tqdm
    print("Downloading ...")
    wget.download(url, f"data/{url.split('/')[-1]}")

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


# def softmax(x):
#     return np.exp(x)/sum(np.exp(x))

def inverse_softmax(p):
    p_inv = torch.zeros(p.shape)

    for i in range(p.size(0)):
      res = least_squares(lambda x: softmax(x)-p[i,:].numpy(),[0]*p.size(1),xtol=1e-15,gtol=1e-15,ftol=1e-15) 

      p_inv[i,:] = torch.tensor(res.x)
    return p_inv

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

num_to_atom = {
    'H':'H',
    'C':'C',
    'O':'O',
    'N':'N',
    'F':'F',
    'M':'??'
}

def red(x):
    # return in RGB order
    # x=0 -> 1, 1, 1 (white)
    # x=1 -> 1, 0, 0 (red)
    return 1., 1. - x, 1. - x

def plotMoleculeAttention(attention,smiles,name,atoms):
  from rdkit import Chem
  from rdkit.Chem import rdDepictor
  from rdkit.Chem.Draw import rdMolDraw2D
  from IPython.display import display, HTML, SVG
  from cairosvg import svg2png
  import tempfile
  import matplotlib.image as mpimage
  import matplotlib.pyplot as plt
  mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
  mc = Chem.Mol(mol.ToBinary())
  try:
      Chem.Kekulize(mol)
  except:
      mc=Chem.Mol(mol.ToBinary())
  rdDepictor.Compute2DCoords(mc)
  threshold = np.mean(attention)
  highlight_atoms=[]
  for j in range(len(attention)):
      if attention[j]>=threshold:
          highlight_atoms.append(j)
  atom_colors = {i: red(e) for i, e in enumerate(attention) if attention[i]>=threshold}
  drawer = rdMolDraw2D.MolDraw2DCairo(10, 10)
  drawer_svg = rdMolDraw2D.MolDraw2DSVG(500,500)
  opts = drawer.drawOptions()
  for i in range(len(atoms)):
      opts.atomLabels[i]=num_to_atom[atoms[i]]
  tm = rdMolDraw2D.PrepareMolForDrawing(mc)
  drawer.DrawMolecule(tm,highlightAtoms=highlight_atoms,highlightAtomColors=atom_colors)
  drawer.FinishDrawing()
  drawer_svg .DrawMolecule(tm,highlightAtoms=highlight_atoms,highlightAtomColors=atom_colors)
  drawer_svg.FinishDrawing()
  svg = drawer_svg.GetDrawingText()


  png = drawer.GetDrawingText()

  with tempfile.NamedTemporaryFile() as tmp:
      tmp.write(png)
      image = mpimage.imread(tmp)
      #image = Image.open(tmp)
      plt.imshow(image, interpolation="bilinear")
      plt.axis('off')
      #image.show()
  #SVG(svg.replace('svg:', ''))
  #svg2png(bytestring=svg, write_to=name+smiles+'.png')



def plot_attention(smiles, atoms,name, attention):
  #for i in range(1,5):
  #att = torch.mean(attention,dim=0).detach().cpu().numpy()
  #attention=(att.sum(axis=0)/len(att.sum(axis=0))).tolist()
  plotMoleculeAttention(attention[0,:].detach().cpu().numpy(), smiles,name, atoms)


def plot_prediction(smiles, atoms, targets, predictions, probabilities):
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
  
  ax=plt.subplot()
  #fig.add_subplot(1,3,1)
  #ax.set_title('Input')
  plot_molecule(mol, highlight_atoms=masked_idx, highlight_atom_colors=masked_color,show=False, ax=ax)
  fig = plt.figure(figsize=(2,2))
  ax=plt.subplot()
    
  #replace the masks with the predictions
  # for i, target,prediction in zip(masked_idx,targets,predictions):
  #   mol.ReplaceAtom(i, rdchem.Atom(int2atom[prediction.item()+1]))
    # if mol.GetBondWithIdx(i).GetIsAromatic():
    #   mol.GetAtomWithIdx(i).SetIsAromatic(1)

  #fig.add_subplot(1,3,2)
  #ax2.set_title('Prediction')
  #coords = plot_molecule(mol,highlight_atoms=masked_idx, highlight_atom_colors=masked_color,show=False, ax=ax, return_idx=masked_idx[0])

  #axins_coords = [0.6, 0.0]
  #axins = ax.inset_axes([axins_coords[0],axins_coords[1],0.3,0.3])
  

  #fig.add_subplot(1,3,3)
  # ax3.set_title('Probability')
  ax.yaxis.set_label_position("right")
  ax.yaxis.tick_right()
  ax.bar([1,2,3,4,5,6,7,8,9,10], probabilities.detach().cpu().numpy()[0,:])
  ax.set_xticks([1,2,3,4,5,6,7,8,9,10])
  ax.set_xticklabels(['H','C','O','N','F','P','S','Cl','Br','I'])
  ax.set_ylim([0,1])
  ax.set_ylabel('Probability')
  #ax.plot([coords.x+10, axins_coords[0]*500], [coords.y+10, (1-axins_coords[1])*1000], color='k', alpha=0.25)
  #ax.plot([coords.x+10, axins_coords[0]*1000], [coords.y+10, (1-axins_coords[1])*1000 - 0.3*1000], color='k', alpha=0.25)
  fig.tight_layout()
  plt.show()

def plot_molecule(molecule, ax, highlight_atoms=[], highlight_atom_colors={}, show=True, inchi=False, return_idx=None):
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
  drawer_svg = draw.rdMolDraw2D.MolDraw2DSVG(500,500)
  draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors)
  draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer_svg, molecule, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors)
  drawer.FinishDrawing()

  png = drawer.GetDrawingText()
  svg = drawer_svg.GetDrawingText()

  with open('images/molecule.svg','w') as f:
    f.write(svg)

  with tempfile.NamedTemporaryFile() as tmp:
      tmp.write(png)
      image = mpimage.imread(tmp)
      #image = Image.open(tmp)
      ax.imshow(image, interpolation="bilinear")
      ax.axis('off')
      #image.show()

  if show:
    plt.show()
  if return_idx is not None:

    return drawer.GetDrawCoords(return_idx)

