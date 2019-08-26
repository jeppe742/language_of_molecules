import pybel
import openbabel
import numpy
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from utils.dataloader import QM9Dataset
import numpy as np
import os
from utils.helpers import download_files

periodic_table = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F'
}

molecules_Adjacency_list = []

if not os.path.exists("data/dsgdb9nsd.xyz"):
    print("Data not found locally")
    download_files()

n_files = len(glob.glob("data/dsgdb9nsd.xyz/*.xyz"))
for k, filename in enumerate(tqdm(glob.glob("data/dsgdb9nsd.xyz/*.xyz"))):

    with open(filename, "r") as file:
        symbols = []
        positions = []
        charges = []

        for row, line in enumerate(file):

            fields = line.strip().split('\t')
            # Each file contains a number of atoms in the first line.
            if row == 0:
                natoms = int(fields[0].rstrip('\n'))
            elif row == 1:
                metadata = fields
            # Then rows of atomic positions and chemical symbols.
            elif row <= natoms + 1:
                symbols.append(fields[0])
                p = [float(j.replace('*^', 'e')) for j in fields[1:4]]
                positions.append(p)
                charges.append(float(fields[4].rstrip('\n').replace('*^',
                                                                    'e')))
            elif row == natoms + 2:
                frequencies = [float(j.rstrip('\n')) for j in fields]
            elif row == natoms + 3:
                # This is the SMILES string.
                smiles = fields[-1]
            elif row == natoms + 4:
                # This is the IbnChI string.
                inchi = [j.rstrip('\n').lstrip('InChI=') for j in fields]

        constants = {
            'rot_a': float(metadata[1]),
            'rot_b': float(metadata[2]),
            'rot_c': float(metadata[3]),
            'mu': float(metadata[4]),
            'alpha': float(metadata[5]),
            'homo': float(metadata[6]),
            'lumo': float(metadata[7]),
            'gap': float(metadata[8]),
            'r2': float(metadata[9]),
            'zpve': float(metadata[10]),
            'u0': float(metadata[11]),
            'u': float(metadata[12]),
            'h': float(metadata[13]),
            'g': float(metadata[14]),
            'cv': float(metadata[15]),
            'charges': charges
        }

        molecule = pybel.readstring('smi', smiles)
        # Hydrogen is normally implicit, but we need them to exist explicitly in the molecule
        molecule.addh()

        n_atoms = len(molecule.atoms)
        Adj = np.zeros((n_atoms, n_atoms), dtype=int)

        np.fill_diagonal(Adj, 1)

        molecule_list = []

        for atom in molecule.atoms:
            atom = atom.OBAtom
            # atoms are indexed from 1, but we want 0 for our matrix
            atom_idx = atom.GetIdx() - 1

            neighbours_idx = [obneighbor.GetIdx() - 1 for obneighbor in openbabel.OBAtomAtomIter(atom)]
            # Set the entries for the adjecency matrix
            for neighbour_idx in neighbours_idx:
                Adj[atom_idx, neighbour_idx] = 1
                Adj[neighbour_idx, atom_idx] = 1
            # add the atom to the list of atoms
            molecule_list.append(periodic_table[atom.GetAtomicNum()])

        # convert list of atoms to numpy array for easier computations later
        molecule_list = np.asarray(molecule_list)

        molecules_Adjacency_list.append([molecule_list, Adj, constants])

print("Splitting data..")
molecules_Adjacency_train, molecules_Adjacency_test = train_test_split(molecules_Adjacency_list, test_size=0.15, random_state=42)
molecules_Adjacency_train, molecules_Adjacency_validation = train_test_split(molecules_Adjacency_train, test_size=0.3, random_state=42)

print("dumping splits..")
pickle.dump(molecules_Adjacency_train, open('data/adjacency_matrix_train.pkl', 'wb'))
pickle.dump(molecules_Adjacency_validation, open('data/adjacency_matrix_validation.pkl', 'wb'))
pickle.dump(molecules_Adjacency_test, open('data/adjacency_matrix_test.pkl', 'wb'))


print("persisting validation sets...")
# Persist the validation and training
for i in range(1, 6):
    val_set_mask = QM9Dataset(data='data/adjacency_matrix_validation.pkl', num_masks=i)
    val_set_mask.save_static_dataset(f'data/val_set_mask{i}.pkl')

    val_set_fake = QM9Dataset(data='data/adjacency_matrix_validation.pkl', num_masks=0, num_fake=i)
    val_set_fake.save_static_dataset(f'data/val_set_fake{i}.pkl')


print("persisting test sets...")
for i in range(1, 6):
    test_set_mask = QM9Dataset(data='data/adjacency_matrix_test.pkl', num_masks=i)
    test_set_mask.save_static_dataset(f'data/test_set_mask{i}.pkl')

    test_set_fake = QM9Dataset(data='data/adjacency_matrix_test.pkl', num_masks=0, num_fake=i)
    test_set_fake.save_static_dataset(f'data/test_set_fake{i}.pkl')
