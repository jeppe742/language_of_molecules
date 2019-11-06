import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from utils.helpers import plot_molecule, ATOMS, atom2int, int2atom

class DataLoader(TorchDataLoader):

    def __init__(self, *args, **kwargs):
        super(DataLoader, self).__init__(*args, **kwargs, collate_fn=molecule_collate_fn)


def pad(sentences, pad_token=0):
    """
    Converts list of sentences into padded sentences in a numpy array

    Args:
        sentences [List] : list of lengths
        pad_token int : padding token
    Returns:
        padded [Numpy Array of size List]: padded sentences of equal length
        lengths: the original length of each sentence
    """
    # Calculat the sequence length of the batch
    lengths = [len(sentence) for sentence in sentences]
    max_len = max(lengths)

    padded = []
    for example in sentences:
        # calculate how much padding is needed for this example
        pads = max_len - len(example)

        # Pad with zeros
        padded.append(np.pad(example, ((0, pads)), 'constant', constant_values=pad_token))

    padded = np.asarray(padded)

    return padded


def molecule_collate_fn(batch):
    return MoleculeBatch(batch)


class MoleculeBatch():

    def __init__(self, molecule_samples):
        self.atoms = []
        self.atoms_num = []
        self.targets = []
        self.targets_num = []
        self.adj = []
        self.adj2 = []
        self.properties = []
        self.target_mask = []
        self.lengths = []
        self.smiles = []
        self.batch_size = len(molecule_samples)

        for sample in molecule_samples:
            self.atoms += [sample.atoms]
            self.atoms_num += [sample.atoms_num]
            self.targets += [sample.targets]
            self.targets_num += [sample.targets_num]
            self.adj += [sample.adj]
            self.adj2 += [sample.adj2]
            self.properties += [sample.properties]
            self.target_mask += [sample.target_mask]
            self.lengths += [sample.length]
            self.smiles += [sample.smiles]

        self.atoms_num = torch.tensor(pad(self.atoms_num))
        self.targets_num = torch.tensor(pad(self.targets_num))
        self.adj = torch.tensor(pad(self.adj))
        self.adj2 = torch.tensor(pad(self.adj2))
        self.target_mask = torch.tensor(pad(self.target_mask), dtype=torch.bool)
        self.lengths = torch.tensor(self.lengths)

        # TODO: include properties in some nice format

    def cuda(self):
        self.atoms_num = self.atoms_num.cuda()
        self.targets_num = self.targets_num.cuda()
        self.adj = self.adj.cuda()
        self.adj2 = self.adj2.cuda()
        self.target_mask = self.target_mask.cuda()
        self.lengths = self.lengths.cuda()


class QM9Dataset(Dataset):
    def __init__(self,
                 data,
                 num_masks=1,
                 num_fake=0,
                 epsilon_greedy=0.0,
                 bond_order=False,
                 static=False,
                 samples_per_molecule=1):
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
        self.static = static
        self.bond_order = bond_order
        self.samples_per_molecule = samples_per_molecule
        self.unique_molecule_buffer = []

        self.corruption = CorruptionTransform(num_masks=num_masks, num_fake=num_fake, epsilon=epsilon_greedy)

        # If data is string, assume we should load a pikle file
        if isinstance(data, str):
            self.datafile = data
            self.data = []
            samples = pickle.load(open(data, 'rb'))
            for (atoms, adj, adj2, properties, smiles) in samples:
                if bond_order:
                    self.data += [MoleculeSample(atoms, adj2, adj2, properties, smiles)]
                else:
                    self.data += [MoleculeSample(atoms, adj, adj2, properties, smiles)]

    def __len__(self):
        if self.num_masks<=self.samples_per_molecule:
            return len(self.data)*self.samples_per_molecule
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[(idx - idx%self.samples_per_molecule)//self.samples_per_molecule]

        sample_corrupted = self.corruption(sample)
        atoms_string = ''.join(sample_corrupted.atoms.tolist())

        while atoms_string in self.unique_molecule_buffer and sample.length>self.num_masks and self.num_masks<=5:
            sample_corrupted = self.corruption(sample)
            atoms_string = ''.join(sample_corrupted.atoms.tolist())

        self.unique_molecule_buffer.append(atoms_string)

        #clear buffer once we have enough samples per molecule
        if len(self.unique_molecule_buffer)==self.samples_per_molecule:
            self.unique_molecule_buffer = []
        return sample_corrupted

    def save_static_dataset(self, filename):
        """
        Generates all the examples in the dataset, and persist them to a file

        Args:
            filename (string): name of the file to save the dataset in
        """
        examples = []

        for i in range(len(self)):
            examples.append(self.__getitem__(i))

        pickle.dump(examples, open(filename, 'wb'))


class MoleculeSample():

    def __init__(self, atoms, adj, adj2, properties, smiles):

        # Set input properties
        self.atoms = atoms

        self.adj = adj
        self.adj2 = adj2
        self.properties = properties
        self.smiles = smiles

        # We need the lenght of the atom for later
        self.length = len(atoms)

        # The targets are not set until we run the corruption transformation
        self.targets = []
        self.target_mask = np.zeros(self.length)

    # We need the atoms in a numeric format, but by linking to the atoms we make sure they stay in sync
    @property
    def atoms_num(self):
        return [atom2int[atom] for atom in self.atoms]

    @property
    def targets_num(self):
        return [atom2int[atom] for atom in self.targets]

    def copy(self):
        """ 
        Create a copy of the molecule.
        This should make sure that the mutable atoms list isn't overridden in the corruption transformation
        """
        return MoleculeSample(self.atoms.copy(), self.adj.copy(), self.adj2.copy(), self.properties.copy(), self.smiles)

    def plot(self):
        plot_molecule(self.smiles)

class CorruptionTransform():
    """
    Apply a corruption to the input molecule,
    by either replacing one or more atoms with the mask token, or with a random atom.

    Args:
        num_masks (int): Number of atoms to mask
        num_fake (int): Number of atoms to replace with random atoms
        epsilon (int): Epsilon parameter in epsilon-greedy scheme

    """

    def __init__(self, num_masks=0, num_fake=0, epsilon=0):
        self.num_masks = num_masks
        self.num_fake = num_fake
        self.epsilon = epsilon

    def pick_epsilon_greedy(self, molecule, num_pick):
        """
        Picks a number of atom indecies, using the epsilon-greedy scheme,
        where we pick a random number of atoms with epsilon probability, and otheruse just use num_pick

        Args:
            molecule (MoleculeSample): Molecule to pick indecies for
            num_pick (int): Number of atoms to pick

        Returns:
            num_pick (int): Number of atoms picked
        """
        num_pick = int(min(molecule.length, num_pick))

        # Epsilon greedy
        r = np.random.rand()
        # Pick input num_masks with probability 1-e
        if r < (1-self.epsilon):
            num_pick = num_pick
        # Pick random number of masks with probability e
        else:
            num_pick = np.random.randint(1, high=molecule.length+1)

        return num_pick

    def corrupt_mask(self, molecule_sample, picked_atoms):
        molecule_sample.atoms[picked_atoms] = 'M'

    def corrupt_fake(self, molecule_sample, picked_atoms):
        old_atoms = molecule_sample.atoms[picked_atoms]

        # Replace the atoms one at a time, by making sure we don't reuse the same atom as exists
        for idx, old_atom in zip(picked_atoms, old_atoms):
            new_atom = np.random.choice([atom for atom in ATOMS[:-1] if atom != old_atom])
            molecule_sample.atoms[idx] = new_atom

    def __call__(self, molecule_sample):
        """
        Applies the transformation to a molecule

        Args:
            molecule_sample (MoleculeSample): molecule we want to corrupt

        Returns:
            molecule_sample (MoleculeSample): Corrupted molecule
        """

        # we are altering the emutable atoms list, so make a copy to avoid overriding
        molecule_sample = molecule_sample.copy()

        num_masks_picked, num_fake_picked = 0, 0

        if self.num_masks > 0:
            num_masks_picked = self.pick_epsilon_greedy(molecule_sample, self.num_masks)
        if self.num_fake > 0:
            num_fake_picked = self.pick_epsilon_greedy(molecule_sample, self.num_fake)

        # Pick the atoms which should either be masked or faked. Make sure we pick at most the number of atoms in the molecule
        num_pick = min(num_masks_picked + num_fake_picked, molecule_sample.length)
        picked_atoms = np.random.choice(molecule_sample.length, num_pick, replace=False)

        # Use the first num_mask_picked indecies for masking, and the remaning for faking.
        picked_mask = picked_atoms[:num_masks_picked]
        picked_fake = picked_atoms[num_masks_picked:]

        # Picked atoms are not in order, so make sure they are by sorting them, when setting the targets
        molecule_sample.targets = molecule_sample.atoms[sorted(picked_atoms)]

        molecule_sample.target_mask[picked_atoms] = 1

        self.corrupt_mask(molecule_sample, picked_mask)
        self.corrupt_fake(molecule_sample, picked_fake)

        return molecule_sample
