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
import glob


periodic_table = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    15:'P',
    16:'S',
    17:'Cl',
    35:'Br',
    53:'I'
}

molecules_Adjacency_list = []
ions = 0

if not os.path.exists("data/250k_rndm_zinc_drugs_clean.smi"):
    print("Data not found locally")
    download_files("https://raw.githubusercontent.com/mkusner/grammarVAE/master/data/250k_rndm_zinc_drugs_clean.smi")
with open('data/250k_rndm_zinc_drugs_clean.smi','r') as file:
    for line in tqdm(file, total=249456):
        smiles = line.strip()

        molecule = Chem.MolFromSmiles(smiles)

        if '+' in smiles or '-' in smiles:
            ions += 1
            #continue

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

        molecules_Adjacency_list.append([molecule_list, Adj, Adj2, {}, smiles])

print(f" {ions} molecules containing ions")


print("Splitting data..")
molecules_Adjacency_train, molecules_Adjacency_test = train_test_split(molecules_Adjacency_list, test_size=0.15, random_state=42)
molecules_Adjacency_train, molecules_Adjacency_validation = train_test_split(molecules_Adjacency_train, test_size=0.15/0.85, random_state=42)

print("dumping splits..")
if not os.path.exists("data/zinc") : os.makedirs("data/zinc")
pickle.dump(molecules_Adjacency_train, open('data/zinc/adjacency_matrix_train.pkl', 'wb'))
pickle.dump(molecules_Adjacency_validation, open('data/zinc/adjacency_matrix_validation.pkl', 'wb'))
pickle.dump(molecules_Adjacency_test, open('data/zinc/adjacency_matrix_test.pkl', 'wb'))

print("Splitting using scaffold")
molecules_train, molecules_validation, molecules_test = scaffold_split(molecules_Adjacency_list, frac_train=0.7, frac_valid=0.15, frac_test=0.15, random_state=42)
pickle.dump(molecules_train, open('data/zinc/adjacency_matrix_train_scaffold.pkl', 'wb'))
pickle.dump(molecules_validation, open('data/zinc/adjacency_matrix_validation_scaffold.pkl', 'wb'))
pickle.dump(molecules_test, open('data/zinc/adjacency_matrix_test_scaffold.pkl', 'wb'))
