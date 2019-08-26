from attention import TransformerLayer,EDGE_ENCODING_TYPE
from torch.nn import ModuleList, Embedding, Linear, ReLU, Softmax
from helpers import length_to_mask
from BaseNetwork import BaseNetwork
import torch

class TransformerModel(BaseNetwork):  
    def __init__(self, 
                num_embeddings=7,
                embedding_dim=64,
                num_layers=1,
                num_heads=1,
                num_classes=5,
                dropout=0.2,
                edge_encoding=EDGE_ENCODING_TYPE.NONE,
                log=False,
                name=None):

        super(TransformerModel, self).__init__(name=name, log=log)

        #TODO: edges embedding countains an embedding for the padding, which we later mask out
        if edge_encoding == EDGE_ENCODING_TYPE.ONEDIMENSIONAL:
            self.edges_embedding = Embedding(3, 1)
        elif edge_encoding == EDGE_ENCODING_TYPE.EMBEDDINGDIM:
            raise NotImplementedError('embedding dim edge encoding does not work yet')
        elif edge_encoding == EDGE_ENCODING_TYPE.RELATIVE_POSITION:
            self.edges_embedding = ModuleList([Embedding(3, embedding_dim),Embedding(3, embedding_dim)])
        else:
            self.edges_embedding = None
        self.embeddings = Embedding(num_embeddings, embedding_dim)
        self.softmax = Softmax(dim=1)   

        self.l_out = Linear(in_features = embedding_dim,out_features = num_classes)
        #self.l_out_constant = Linear(in_features = embedding_dim, out_features = 1)
        self.attention_layers = ModuleList(
            [TransformerLayer(in_features=embedding_dim, hidden_dim=embedding_dim, num_heads=num_heads, dropout=dropout, edge_encoding=edge_encoding) for _ in range(num_layers)]
        )
       
    def forward(self, batch):
        out={}
     
        atoms = batch.atoms
        if isinstance(atoms, tuple):
            atoms, lengths = atoms
            
        adj = batch.adj
        target_mask = batch.target_mask

        #create mask
        mask = length_to_mask(lengths, dtype=torch.uint8) #[batch_size, molecule_size]
 
        # get embeddings
        x = self.embeddings(atoms) #[batch_size, molecule_size, embedding_dim]

        for i, attention_layer in enumerate(self.attention_layers):
            x, out[f'attention_weights_{i}'] = attention_layer(x, mask, adj, self.edges_embedding)

        #out['constant'] = self.l_out_constant(x[(1-mask)])
        out['out'] = x = self.l_out(x[target_mask,:])
        
        out['prediction']=torch.argmax(self.softmax(x),dim=1)

        return out


if __name__ == "__main__":
    import argparse
    from dataloader import QM9AtomDataset
    from torchtext.data import Iterator
    from torch.optim import Adam
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs',  default=2, type=int)
    parser.add_argument('--batch_size', default=248, type=int)
    parser.add_argument('--edge_encoding', default=4, type=int)
    parser.add_argument('--epsilon_greedy', default=0.2, type=float)
    parser.add_argument('--num_masks', default=1, type=int)
    parser.add_argument('--num_fake', default=0, type=int)
    parser.add_argument('--num_same', default=0, type=int)
    #parser.add_argument('--constant_prediction', default = False, type=bool)
    parser.add_argument('--name_postfix',default='', type=str)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    training = QM9AtomDataset(data = 'data/adjacency_matrix_train.pkl',
                      num_masks=args.num_masks,
                      epsilon_greedy=args.epsilon_greedy,
                      num_fake=args.num_fake,
                      num_same=args.num_same)

    train_iter = Iterator(
                        training,
                        batch_size =args.batch_size ,
                        device=device)
    

    #Create multiple validation iterators, one for 25, 50 and 75% masked atoms
    val_iters = []
    if args.num_fake==0:
        for validation_file in ['data/val_set_mask1.pkl','data/val_set_mask2.pkl','data/val_set_mask3.pkl','data/val_set_mask4.pkl','data/val_set_mask5.pkl']:

            val_set = QM9AtomDataset(data=validation_file, static=True)
            val_iter = Iterator(
                        val_set,
                        batch_size = args.batch_size,
                        device=device)
            val_iters.append(val_iter)
            
    if args.num_masks==0:
        for validation_file in ['data/val_set_fake1.pkl','data/val_set_fake2.pkl','data/val_set_fake3.pkl','data/val_set_fake4.pkl','data/val_set_fake5.pkl']:
            
            val_set = QM9AtomDataset(data=validation_file, static=True)
            val_iter = Iterator(
                        val_set,
                        batch_size = args.batch_size,
                        device=device)
            val_iters.append(val_iter)

    transformerModel = TransformerModel(num_layers=args.num_layers,
                                        num_heads=args.num_heads, 
                                        embedding_dim=args.embedding_dim,
                                        dropout=args.dropout,
                                        edge_encoding=args.edge_encoding,
                                        log=True,
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
    
    optimizer_fun = lambda param:Adam(param, lr=args.lr)
    transformerModel.train_network(train_iter, val_iters, num_epochs=args.num_epochs, eval_after_epochs=1, 
                 log_after_epochs=1, optimizer_fun=optimizer_fun, save_model=True)
    
    # transformerModel.confusion_matrix(class_labels=['H','C','O','N','F'])
    
    
    #transformerModel.plot_PP_per_num_atoms(val_iters)
    #transformerModel.plot_accuracy_per_num_atoms(val_iters)
    #transformerModel.plot_attention(next(iter(val_iter)))
    

    
