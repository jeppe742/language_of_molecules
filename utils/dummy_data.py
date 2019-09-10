from torch.utils.data import Dataset
from utils.dataloader import CorruptionTransform, MoleculeSample
import numpy as np
import rdkit.Chem.rdchem as rdchem
import rdkit.Chem as chem

periodic_table = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F'
}

class DummyDataset(Dataset):
    def __init__(self,
                 num_masks=1,
                 num_fake=0,
                 epsilon_greedy=0.0,
                 num_classes = 2,
                 num_samples = 1000,
                 max_length = 30,
                 ambiguity=False,
                 num_bondtypes=1):
        """Create a dataset of graphs from the QM9 data.

        Arg:
            data (string): file path to a pickle file
            num_masks (int): Number of atoms to mask in each molecule
            num_fake (int): Number of atoms to fake in each molecule
            epsilon_greedy (float): epsilon parameter in the epsilon-greedy scheme for selecting number of corrupted atoms
        """

        self.num_masks = num_masks
        self.num_fake = num_fake
        self.epsilon_greedy = epsilon_greedy
        self.num_classes = num_classes
        self.num_samples = num_samples

        self.molecule_generator = MoleculeGenerator(num_classes=num_classes, 
                                                    max_length=max_length, 
                                                    ambiguity=ambiguity,
                                                    num_bondtypes=num_bondtypes)

        self.corruption = CorruptionTransform(num_masks=num_masks, num_fake=num_fake, epsilon=epsilon_greedy)


        self.data = []


        for i in range(self.num_samples):
            molecule = self.molecule_generator.generate_molecule()
            molecule = chem.AddHs(molecule)


            Adj = chem.rdmolops.GetAdjacencyMatrix(molecule)
            atoms = np.asarray([periodic_table[atom.GetAtomicNum()] for atom in molecule.GetAtoms()])
            smiles = chem.MolToSmiles(molecule)
            self.data += [MoleculeSample(atoms, Adj, {}, smiles)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.corruption:
            sample = self.corruption(sample)

        return sample

class MoleculeGenerator():

    def __init__(self, num_classes=2, max_length=30, ambiguity=False, num_bondtypes=1):
        self.num_classes = num_classes
        self.max_length = max_length
        self.atom2bonds = {'H':1,'O':2,'N':3,'C':4,'F':1}
        self.atoms = ['H','O','F','N','C'][:(num_classes+1)] if ambiguity else ['H','O','N','C'][:num_classes]
        self.bonds = [1, 2, 3][:num_bondtypes]

    def generate_molecule(self):

        mol = rdchem.RWMol()
        
        atom = np.random.choice(self.atoms)
        num_free_bonds = self.atom2bonds[atom]

        idx = mol.AddAtom(rdchem.Atom(atom))
        
        mol = self.add_atom(mol, idx,num_free_bonds)


        mol = mol.GetMol()
        chem.Kekulize(mol)    
        return mol



    def add_atom(self, mol, atomidx, num_free_bonds):
        
        #Cut off recursion. We will later replace freebonds with hydrogen
        if mol.GetNumAtoms() >= self.max_length:
            return mol

        while num_free_bonds > 0:

            #available_atoms = self.atoms[:num_free_bonds]
            #Pick an atom and add it to the molecule
            atom = np.random.choice(self.atoms)
            new_idx = mol.AddAtom(rdchem.Atom(atom))

            #Pick the type of bond, by making sure that we have enough free electrons to create the bond
            bondtype = np.random.choice(self.bonds[:min(num_free_bonds,self.atom2bonds[atom])])


            #connect it to the atomidx we are given
            mol.AddBond(atomidx, new_idx, rdchem.BondType(bondtype))
            #Find the number of bonds the new atom has, minus the only to the original atom
            new_num_free_bonds = self.atom2bonds[atom] - bondtype
            #Call recursively on the new atom
            self.add_atom(mol, new_idx, new_num_free_bonds)
            num_free_bonds -= bondtype
        return mol
