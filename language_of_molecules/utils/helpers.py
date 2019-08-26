import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch


def download_files():
    import wget
    import tarfile
    url = "https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/3195389/dsgdb9nsd.xyz.tar.bz2"
    print("Downloading ...")
    wget.download(url, "data/dsgdb9nsd.xyz.tar.bz2")

    print("\nunzipping ...")
    tar = tarfile.open("data/dsgdb9nsd.xyz.tar.bz2")
    tar.extractall(path="data/dsgdb9nsd.xyz")


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


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


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
