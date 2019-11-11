import numpy
import rdkit.Chem as Chem
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from utils.dataloader import QM9Dataset
import numpy as np
import os
from utils.helpers import download_files, scaffold_split, plot_molecule



periodic_table = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F'
}

molecules_Adjacency_list = []
ions = 0

if not os.path.exists("data/qm9.csv"):
    print("Data not found locally")
    download_files("http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm9.csv")

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

        if '+' in smiles or '-' in smiles:
            ions += 1
            continue
        molecule = Chem.MolFromSmiles(smiles)

        #convert aromatic bonds to single/double
        Chem.Kekulize(molecule)
        # Hydrogen is normally implicit, but we need them to exist explicitly in the molecule
        molecule = Chem.AddHs(molecule)

        molecule_list = []

        for atom in molecule.GetAtoms():
            molecule_list.append(periodic_table[atom.GetAtomicNum()])
        
        Adj = Chem.rdmolops.GetAdjacencyMatrix(molecule)


        Adj2 = np.zeros(Adj.shape, dtype=np.int32)
        for bond in molecule.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            Adj2[start,end] = int(bond_type)
            Adj2[end,start] = int(bond_type)
        # convert list of atoms to numpy array for easier computations later
        molecule_list = np.asarray(molecule_list)

        molecules_Adjacency_list.append([molecule_list, Adj, Adj2, constants, smiles])
        
print(f"{ions} molecules containing ions")
print("Splitting data..")
molecules_Adjacency_train, molecules_Adjacency_test = train_test_split(molecules_Adjacency_list, test_size=0.15, random_state=42)
molecules_Adjacency_train, molecules_Adjacency_validation = train_test_split(molecules_Adjacency_train, test_size=0.15/0.85, random_state=42)

print("dumping splits..")
if not os.path.exists("data/qm9") : os.makedirs("data/qm9")
pickle.dump(molecules_Adjacency_train, open('data/qm9/adjacency_matrix_train.pkl', 'wb'))
pickle.dump(molecules_Adjacency_validation, open('data/qm9/adjacency_matrix_validation.pkl', 'wb'))
pickle.dump(molecules_Adjacency_test, open('data/qm9/adjacency_matrix_test.pkl', 'wb'))

print("Splitting using scaffold")
molecules_train, molecules_validation, molecules_test = scaffold_split(molecules_Adjacency_list, frac_train=0.7, frac_valid=0.15, frac_test=0.15, random_state=42)
pickle.dump(molecules_train, open('data/qm9/adjacency_matrix_train_scaffold.pkl', 'wb'))
pickle.dump(molecules_validation, open('data/qm9/adjacency_matrix_validation_scaffold.pkl', 'wb'))
pickle.dump(molecules_test, open('data/qm9/adjacency_matrix_test_scaffold.pkl', 'wb'))
