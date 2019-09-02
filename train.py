from torch.optim import Adam
import torch
import argparse
from utils.dataloader import QM9Dataset, DataLoader
from layers.transformer import TransformerModel
from layers.bagofwords import BagOfWordsModel, BagOfWordsType
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_heads', default=3, type=int)
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_epochs',  default=10, type=int)
parser.add_argument('--batch_size', default=248, type=int)
parser.add_argument('--edge_encoding', default=1, type=int)
parser.add_argument('--epsilon_greedy', default=0.2, type=float)
parser.add_argument('--num_masks', default=1, type=int)
parser.add_argument('--num_fake', default=0, type=int)
parser.add_argument('--num_same', default=0, type=int)
parser.add_argument('--name_postfix', default='', type=str)
parser.add_argument('--use_cuda', default=True, type=bool)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--model',choices=['BoN','BoA','Transformer'], default='Transformer')
args = parser.parse_args()


training = QM9Dataset(data='data/adjacency_matrix_train.pkl',
                      num_masks=args.num_masks,
                      epsilon_greedy=args.epsilon_greedy,
                      num_fake=args.num_fake)

train_dl = DataLoader(
    training,
    batch_size=args.batch_size)


# Create multiple validation dlators, one for 25, 50 and 75% masked atoms
val_dls = []
if args.num_fake == 0:
    for masks in range(1, 6):

        val_set = QM9Dataset(data='data/adjacency_matrix_validation.pkl', num_masks=masks)
        val_dl = DataLoader(
            val_set,
            batch_size=args.batch_size)
        val_dls.append(val_dl)

if args.num_masks == 0:
    for fakes in range(1, 6):

        val_set = QM9Dataset(data='data/adjacency_matrix_validation.pkl', num_fake=fakes)
        val_dl = DataLoader(
            val_set,
            batch_size=args.batch_size)
        val_dls.append(val_dl)


if args.model =='Transformer':
        
    model = TransformerModel(num_layers=args.num_layers,
                                        num_heads=args.num_heads,
                                        embedding_dim=args.embedding_dim,
                                        dropout=args.dropout,
                                        edge_encoding=args.edge_encoding,
                                        use_cuda=args.use_cuda,
                                        name=(
                                            "Transformer"
                                            f"_num_masks={args.num_masks}"
                                            f"_num_fake={args.num_fake}"
                                            f"_num_same={args.num_same}"
                                            f"_num_layers={args.num_layers}"
                                            f"_num_heads={args.num_heads}"
                                            f"_embedding_dim={args.embedding_dim}"
                                            f"_dropout={args.dropout}"
                                            f"_lr={args.lr}"
                                            f"_edge_encoding={args.edge_encoding}"
                                            f"_epsilon_greedy={args.epsilon_greedy}"
                                            f"{args.name_postfix}"
                                        )
                                    )
elif args.model == 'BoA':
    model = BagOfWordsModel(num_layers=args.num_layers,
                                        embedding_dim=args.embedding_dim,
                                        BagOfWordsType=BagOfWordsType.ATOMS,
                                        use_cuda=args.use_cuda,
                                        name=(
                                            "BagOfWords"
                                            f"_num_masks={args.num_masks}"
                                            f"_num_fake={args.num_fake}"
                                            f"_num_same={args.num_same}"
                                            f"_num_layers={args.num_layers}"
                                            f"_embedding_dim={args.embedding_dim}"
                                            f"_lr={args.lr}"
                                            f"_epsilon_greedy={args.epsilon_greedy}"
                                            f"_bow_type={BagOfWordsType.ATOMS}"
                                            f"{args.name_postfix}"
                                        )
                        )
elif args.model == 'BoN':
    model = BagOfWordsModel(num_layers=args.num_layers,
                                        embedding_dim=args.embedding_dim,
                                        BagOfWordsType=BagOfWordsType.NEIGHBOURS,
                                        use_cuda=args.use_cuda,
                                        name=(
                                            "BagOfWords"
                                            f"_num_masks={args.num_masks}"
                                            f"_num_fake={args.num_fake}"
                                            f"_num_same={args.num_same}"
                                            f"_num_layers={args.num_layers}"
                                            f"_embedding_dim={args.embedding_dim}"
                                            f"_lr={args.lr}"
                                            f"_epsilon_greedy={args.epsilon_greedy}"
                                            f"_bow_type={BagOfWordsType.NEIGHBOURS}"
                                            f"{args.name_postfix}"
                                        )
                        )

def optimizer_fun(param): return Adam(param, lr=args.lr)


if not args.debug:
    wandb.init(project="language-of-molecules", name=model.name)
    wandb.config.update(args)
    wandb.watch(model)

model.train_network(train_dl, val_dls, num_epochs=args.num_epochs, eval_after_epochs=1,
                               log_after_epochs=1, optimizer_fun=optimizer_fun, save_model=True)
