import rdkit.Chem as Chem
import glob
from tqdm import tqdm
import pickle
from utils.dataloader import QM9Dataset
import numpy as np
import os
from utils.helpers import download_files
import pandas as pd


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

if not os.path.exists("data/250k_rndm_zinc_drugs_clean_3.csv"):
    print("Data not found locally")
    download_files("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")

values = []
smiles = []
with open('data/250k_rndm_zinc_drugs_clean_3.csv') as f:
    next(f)
    for line in f:
        info = line.split('",')
        if len(info)>1:
            values.append(info[1].strip().split(','))
        else:
            smiles.append(info[0].strip('"').strip())
data =np.asarray(values,dtype=np.float)
data = pd.DataFrame({'smiles':smiles,'LogP':data[:,0],'QED':data[:,1],'SAS':data[:,2]})


test_samples=data.sample(frac=0.2)
remaining_samples = data.loc[~data.index.isin(test_samples.index)]
train_samples = remaining_samples[remaining_samples.QED>0.75]

for name, dataset in zip(['train','test'],[train_samples, test_samples]): 
    for i, sample in tqdm(dataset.iterrows(),total=len(dataset)):
        smiles = sample.smiles
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

        charges = [atom.GetFormalCharge() for atom in molecule.GetAtoms()]
        num_neighbours = Adj2.sum(axis=1)
        molecules_Adjacency_list.append([molecule_list, Adj, Adj2, {'QED':sample.QED,'SAS':sample.SAS,'LogP':sample.LogP}, smiles, charges, num_neighbours])
    
    if not os.path.exists("data/zinc_properties") : os.makedirs("data/zinc_properties")
    pickle.dump(molecules_Adjacency_list,open(f'data/zinc_properties/adjacency_matrix_{name}.pkl','wb'))
print(f" {ions} molecules containing ions")
