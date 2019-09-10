import numpy
import rdkit.Chem as Chem
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from utils.dataloader import QM9Dataset
import numpy as np
import os
from utils.helpers import download_files, scaffold_split



periodic_table = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F'
}

molecules_Adjacency_list = []

if not os.path.exists("data/qm9.csv"):
    print("Data not found locally")
    download_files()

with open('data/qm9.csv','r') as file:
    #skip header
    next(file)
    for line in tqdm(file, total=133886):
        symbols = []
        positions = []
        charges = []

      
        fields = line.strip().split(',')

        smiles = fields[1]

        constants = {
            'rot_a': float(fields[2]),
            'rot_b': float(fields[3]),
            'rot_c': float(fields[4]),
            'mu': float(fields[5]),
            'alpha': float(fields[6]),
            'homo': float(fields[7]),
            'lumo': float(fields[8]),
            'gap': float(fields[9]),
            'r2': float(fields[10]),
            'zpve': float(fields[11]),
            'u0': float(fields[12]),
            'u': float(fields[13]),
            'h': float(fields[14]),
            'g': float(fields[15]),
            'cv': float(fields[16])
        }


        molecule = Chem.MolFromSmiles(smiles)
        # Hydrogen is normally implicit, but we need them to exist explicitly in the molecule
        molecule = Chem.AddHs(molecule)

        molecule_list = []

        for atom in molecule.GetAtoms():
            molecule_list.append(periodic_table[atom.GetAtomicNum()])

        Adj = Chem.rdmolops.GetAdjacencyMatrix(molecule)

        # convert list of atoms to numpy array for easier computations later
        molecule_list = np.asarray(molecule_list)

        molecules_Adjacency_list.append([molecule_list, Adj, constants, smiles])

print("Splitting data..")
molecules_Adjacency_train, molecules_Adjacency_test = train_test_split(molecules_Adjacency_list, test_size=0.15, random_state=42)
molecules_Adjacency_train, molecules_Adjacency_validation = train_test_split(molecules_Adjacency_train, test_size=0.15/0.85, random_state=42)

print("dumping splits..")
pickle.dump(molecules_Adjacency_train, open('data/adjacency_matrix_train.pkl', 'wb'))
pickle.dump(molecules_Adjacency_validation, open('data/adjacency_matrix_validation.pkl', 'wb'))
pickle.dump(molecules_Adjacency_test, open('data/adjacency_matrix_test.pkl', 'wb'))

#print("Splitting using scaffold")
molecules_train, molecules_validation, molecules_test = scaffold_split(molecules_Adjacency_list, frac_train=0.7, frac_valid=0.15, frac_test=0.15, random_state=42)
pickle.dump(molecules_train, open('data/adjacency_matrix_train_scaffold.pkl', 'wb'))
pickle.dump(molecules_validation, open('data/adjacency_matrix_validation_scaffold.pkl', 'wb'))
pickle.dump(molecules_test, open('data/adjacency_matrix_test.pkl_scaffold', 'wb'))
