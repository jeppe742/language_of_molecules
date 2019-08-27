from torch.nn import ModuleList, Embedding, Linear, ReLU, Softmax
from utils.helpers import length_to_mask
from layers.BaseNetwork import BaseNetwork
import torch


class BagOfWordsType:
    ATOMS = 1
    NEIGHBOURS = 2
    GRAPH = 3


class BagOfWordsLayer(BaseNetwork):
    def __init__(self, embedding_dim=64, BagOfWordsType=BagOfWordsType.ATOMS):
        super(BagOfWordsLayer, self).__init__()

        self.linear_out = Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.type = BagOfWordsType

        self.relu = ReLU()

    def forward(self, atoms, mask, adj):

        batch_size, molecule_size, embedding_dim = atoms.shape
        mask = mask.unsqueeze(2).permute(0, 2, 1)  # [batch_size, molecule_size, molecule_size]

        weights = torch.ones(batch_size, molecule_size, molecule_size, device=atoms.device)

        weights.masked_fill(mask, 0)

        if self.type == BagOfWordsType.NEIGHBOURS:
            # Mask all but neighbours
            weights.masked_fill(1-adj, 0)

        bow = torch.matmul(weights, atoms)  # [batch_size, molecule_size, embedding_dim]

        return self.relu(self.linear_out(bow))


class BagOfWordsModel(BaseNetwork):
    def __init__(self,
                 num_embeddings=7,
                 embedding_dim=64,
                 num_layers=1,
                 num_classes=5,
                 name=None,
                 BagOfWordsType=BagOfWordsType.ATOMS):

        super(BagOfWordsModel, self).__init__(name=name)

        self.embedding_dim = embedding_dim
        self.embeddings = Embedding(num_embeddings, embedding_dim)
        self.softmax = Softmax(dim=1)

        self.l_out = Linear(in_features=embedding_dim, out_features=num_classes)

        self.bow_layers = ModuleList(
            [BagOfWordsLayer(embedding_dim=embedding_dim, BagOfWordsType=BagOfWordsType) for _ in range(num_layers)]
        )

    def forward(self, batch):
        out = {}

        atoms = batch.atoms
        if isinstance(atoms, tuple):
            atoms, lengths = atoms

        batch_size, molecule_size = atoms.shape
        target_mask = batch.target_mask
        adj = batch.adj

        # create mask
        mask = length_to_mask(lengths, dtype=torch.bool)  # [batch_size, molecule_size]

        # get embeddings
        x = self.embeddings(atoms)  # [batch_size, molecule_size, embedding_dim]

        for bow_layer in self.bow_layers:
            x = bow_layer(x, mask, adj)

        out['out'] = x = self.l_out(x[target_mask, :])

        out['prediction'] = torch.argmax(self.softmax(x), dim=1)

        return out


if __name__ == "__main__":
    import argparse
    from dataloader import QM9AtomDataset
    from torchtext.data import Iterator
    from torch.optim import Adam
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs',  default=30, type=int)
    parser.add_argument('--batch_size', default=248, type=int)
    parser.add_argument('--epsilon_greedy', default=0.2, type=float)
    parser.add_argument('--num_masks', default=1, type=int)
    parser.add_argument('--num_fake', default=0, type=int)
    parser.add_argument('--num_same', default=0, type=int)
    parser.add_argument('--bow_type', default=BagOfWordsType.ATOMS, type=int)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    training = QM9AtomDataset(data='data/adjacency_matrix_train.pkl',
                              num_masks=args.num_masks,
                              epsilon_greedy=args.epsilon_greedy,
                              num_fake=args.num_fake,
                              num_same=args.num_same)

    train_iter = Iterator(
        training,
        batch_size=args.batch_size,
        device=device)

    # Create multiple validation iterators, one for 25, 50 and 75% masked atoms
    val_iters = []
    if args.num_fake == 0:
        for validation_file in ['data/val_set_mask1.pkl', 'data/val_set_mask2.pkl', 'data/val_set_mask3.pkl', 'data/val_set_mask4.pkl', 'data/val_set_mask5.pkl']:

            val_set = QM9AtomDataset(data=validation_file, static=True)
            val_iter = Iterator(
                val_set,
                batch_size=args.batch_size,
                device=device)
            val_iters.append(val_iter)

    if args.num_masks == 0:
        for validation_file in ['data/val_set_fake1.pkl', 'data/val_set_fake2.pkl', 'data/val_set_fake3.pkl', 'data/val_set_fake4.pkl', 'data/val_set_fake5.pkl']:

            val_set = QM9AtomDataset(data=validation_file, static=True)
            val_iter = Iterator(
                val_set,
                batch_size=args.batch_size,
                device=device)
            val_iters.append(val_iter)

    bagOfWordsModel = BagOfWordsModel(num_layers=args.num_layers,
                                      embedding_dim=args.embedding_dim,
                                      BagOfWordsType=args.bow_type,
                                      name=(
                                          "BagOfWords"
                                          f"_num_masks={args.num_masks}"
                                          f"_num_fake={args.num_fake}"
                                          f"_num_same={args.num_same}"
                                          f"_num_layers={args.num_layers}"
                                          f"_embedding_dim={args.embedding_dim}"
                                          f"_lr={args.lr}"
                                          f"_epsilon_greedy={args.epsilon_greedy}"
                                          f"_bow_type={args.bow_type}"
                                      )
                                      )

    def optimizer_fun(param): return Adam(param, lr=args.lr)
    bagOfWordsModel.train_network(train_iter, val_iters, num_epochs=args.num_epochs, eval_after_epochs=1,
                                  log_after_epochs=2, optimizer_fun=optimizer_fun, save_model=True)

    # transformerModel.confusion_matrix(class_labels=['H','C','O','N','F'])

    bagOfWordsModel.plot_PP_per_num_atoms(val_iters)
    bagOfWordsModel.plot_accuracy_per_num_atoms(val_iters)
    # transformerModel.plot_attention(next(iter(val_iter)))
