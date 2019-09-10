from utils.dataloader import MoleculeSample, CorruptionTransform
import numpy as np


def test_corruption_override_mask():
    corruption = CorruptionTransform(num_masks=1)

    molecule = MoleculeSample(np.array(['H', 'C', 'C', 'H', 'O', 'N', 'N', 'H']), np.eye(8), {'a': 1},'')

    _ = corruption(molecule)

    assert np.array_equal(molecule.atoms, np.array(['H', 'C', 'C', 'H', 'O', 'N', 'N', 'H']))
    assert np.array_equal(molecule.atoms_num, np.array([1, 2, 2, 1, 3, 4, 4, 1]))
    assert np.array_equal(molecule.adj, np.eye(8))
    assert molecule.properties == {'a': 1}


def test_corruption_override_fake():
    corruption = CorruptionTransform(num_fake=1)

    molecule = MoleculeSample(np.array(['H', 'C', 'C', 'H', 'O', 'N', 'N', 'H']), np.eye(8), {'a': 1},'')

    _ = corruption(molecule)

    assert np.array_equal(molecule.atoms, np.array(['H', 'C', 'C', 'H', 'O', 'N', 'N', 'H']))
    assert np.array_equal(molecule.atoms_num, np.array([1, 2, 2, 1, 3, 4, 4, 1]))
    assert np.array_equal(molecule.adj, np.eye(8))
    assert molecule.properties == {'a': 1}


def test_corruption_override_both():
    corruption = CorruptionTransform(num_fake=2, num_masks=2)

    molecule = MoleculeSample(np.array(['H', 'C', 'C', 'H', 'O', 'N', 'N', 'H']), np.eye(8), {'a': 1},'')

    _ = corruption(molecule)

    assert np.array_equal(molecule.atoms, np.array(['H', 'C', 'C', 'H', 'O', 'N', 'N', 'H']))
    assert np.array_equal(molecule.atoms_num, np.array([1, 2, 2, 1, 3, 4, 4, 1]))
    assert np.array_equal(molecule.adj, np.eye(8))
    assert molecule.properties == {'a': 1}


def test_corruption_mask1():
    corruption = CorruptionTransform(num_masks=1)

    molecule = MoleculeSample(np.array(['H', 'C', 'C', 'H', 'O', 'N', 'N', 'H']), np.eye(8), {'a': 1},'')

    np.random.seed(0)
    new_sample = corruption(molecule)

    assert np.array_equal(new_sample.atoms, np.array(['H', 'C', 'C', 'H', 'O', 'N', 'M', 'H']))
    assert np.array_equal(new_sample.atoms_num, np.array([1, 2, 2, 1, 3, 4, 6, 1]))
    assert np.array_equal(new_sample.targets, np.array(['N']))
    assert np.array_equal(new_sample.target_mask, np.array([0, 0, 0, 0, 0, 0, 1, 0]))


def test_corruption_mask2():
    corruption = CorruptionTransform(num_masks=2)

    molecule = MoleculeSample(np.array(['H', 'C', 'C', 'H', 'O', 'N', 'N', 'H']), np.eye(8), {'a': 1},'')

    np.random.seed(0)
    new_sample = corruption(molecule)

    assert np.array_equal(new_sample.atoms, np.array(['H', 'C', 'M', 'H', 'O', 'N', 'M', 'H']))
    assert np.array_equal(new_sample.atoms_num, np.array([1, 2, 6, 1, 3, 4, 6, 1]))
    assert np.array_equal(new_sample.targets, np.array(['C', 'N']))
    assert np.array_equal(new_sample.target_mask, np.array([0, 0, 1, 0, 0, 0, 1, 0]))


def test_corruption_fake1():
    corruption = CorruptionTransform(num_fake=1)

    molecule = MoleculeSample(np.array(['H', 'H', 'H', 'H', 'H']), np.eye(5), {'a': 1},'')

    np.random.seed(0)
    new_sample = corruption(molecule)

    assert np.array_equal(new_sample.atoms, np.array(['H', 'H', 'H', 'H', 'O']))
    assert np.array_equal(new_sample.atoms_num, np.array([1, 1, 1, 1, 3]))
    assert np.array_equal(new_sample.targets, np.array(['H']))
    assert np.array_equal(new_sample.target_mask, np.array([0, 0, 0, 0, 1]))


def test_corruption_fake2():
    corruption = CorruptionTransform(num_fake=2)

    molecule = MoleculeSample(np.array(['H', 'H', 'O', 'H', 'H']), np.eye(5), {'a': 1},'')

    np.random.seed(0)
    new_sample = corruption(molecule)

    assert np.array_equal(new_sample.atoms, np.array(['H', 'H', 'N', 'H', 'O']))
    assert np.array_equal(new_sample.atoms_num, np.array([1, 1, 4, 1, 3]))
    assert np.array_equal(new_sample.targets, np.array(['O', 'H']))
    assert np.array_equal(new_sample.target_mask, np.array([0, 0, 1, 0, 1]))


def test_corruption_mask2_fake2():
    corruption = CorruptionTransform(num_fake=2, num_masks=2)

    molecule = MoleculeSample(np.array(['H', 'N', 'N', 'H', 'H']), np.eye(5), {'a': 1},'')

    np.random.seed(0)
    new_sample = corruption(molecule)

    assert np.array_equal(new_sample.atoms, np.array(['M', 'C', 'M', 'H', 'N']))
    assert np.array_equal(new_sample.atoms_num, np.array([6, 2, 6, 1, 4]))
    assert np.array_equal(new_sample.targets, np.array(['H', 'N', 'N', 'H']))
    assert np.array_equal(new_sample.target_mask, np.array([1, 1, 1, 0, 1]))


def test_corruption_epsilon():
    corruption = CorruptionTransform(num_masks=1, epsilon=1)

    molecule = MoleculeSample(np.array(['H', 'H', 'N', 'H', 'H']), np.eye(5), {'a': 1},'')

    np.random.seed(3)
    new_sample = corruption(molecule)

    assert np.array_equal(new_sample.atoms, np.array(['H', 'M', 'M', 'H', 'H']))
    assert np.array_equal(new_sample.atoms_num, np.array([1, 6, 6, 1, 1]))
    assert np.array_equal(new_sample.targets, np.array(['H', 'N']))
    assert np.array_equal(new_sample.target_mask, np.array([0, 1, 1, 0, 0]))
