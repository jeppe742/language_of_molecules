import numpy as np
from helpers import softmax
from copy import deepcopy

class UniformClassifier():
    def __init__(self):
        pass

    def fit(self, x, y):
        # Find the number of unique elements. padding is -1, so disregard this
        self.V = len(np.unique(y[y >= 0]))

        self.P = [1/self.V]*self.V

    def predict(self, x, return_prob=True):
        predictions = []
        probabilities = []
        for _ in x:
            predictions.append(np.argmax(self.P))
            probabilities.append(self.P)

        if return_prob:
            return predictions, probabilities
        return predictions


class UnigramClassifier():

    def fit(self, x, y):
        _, counts = np.unique(y[y >= 0], return_counts=True)
        self.V = len(counts)

        self.P = counts/sum(counts)

    def predict(self, x, return_prob=True):
        predictions = []
        probabilities = []
        for _ in x:
            predictions.append(np.argmax(self.P))
            probabilities.append(self.P)

        if return_prob:
            return predictions, probabilities
        return predictions


class NaiveBayesClassifier():
    def __init__(self, prior=None, binary=False, smooth=0):
        # Set prior P_c
        self.P_c = prior
        self.binary = binary
        self.smooth = smooth

    def binarize(self, molecules):
        molecules_binary = []

        for molecule in molecules:
            molecule_binary = []
            for atom in molecule:

                if atom not in molecule_binary and atom >= 0:
                    molecule_binary.append(atom)

            # Add missing padding, to make it a fixed length molecule
            for missing_padding in range(self.V-len(molecule_binary)):
                molecule_binary.append(-1)

            molecules_binary.append(molecule_binary)
        return np.asarray(molecules_binary)

    def fit(self, molecules, labels):

        molecules = deepcopy(molecules)
        labels = deepcopy(labels)

        vals = np.unique(molecules[molecules >= 0])
        self.V = len(vals)

        if self.binary:
            molecules = self.binarize(molecules)

        # Count the label distribution
        vals, counts = np.unique(labels, return_counts=True)
        self.classes = vals.astype(np.int)
        self.num_classes = len(vals)

        # If no prior is given, use the unigram
        if self.P_c is None:
            self.P_c = counts/sum(counts)

        p_w_given_c = np.zeros((self.V, self.num_classes))

        # Loop over all the classes
        for c in self.classes:
            # Count the molecules, that have label c
            molecules_tmp = molecules[labels == c, :]
            counts, _ = np.histogram(molecules_tmp[molecules_tmp >= 0], bins=range(self.V+1))
            p_w_given_c[:, c] = counts + self.smooth
        # normalize columns wise (over each class)
        self.p_w_given_c = p_w_given_c/(np.sum(p_w_given_c, axis=0) + self.V*self.smooth)

    def predict(self, molecules, return_prob=True):

        molecules = deepcopy(molecules)

        # Binarize input
        if self.binary:
            molecules = self.binarize(molecules)

        predictions = []
        probabilities = []

        for molecule in molecules:
            P = [0]*self.num_classes
            for c in self.classes:
                # add log prior
                P[c] = np.log(self.P_c[c])

                for atom in molecule:
                    if atom >= 0:
                        # calculate log likelihood. Molecules has pad as index 0, so shift it one
                        P[c] += np.log(self.p_w_given_c[int(atom), c])

            # normalize the posterior distribution, using softmax
            P = softmax(P)
            predictions.append(np.argmax(P))
            probabilities.append(P)
        if return_prob:
            return predictions, probabilities
        return predictions
