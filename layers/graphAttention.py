from attention import GraphAttentionLayer, EDGE_ENCODING_TYPE
from torch.nn import ModuleList, Embedding, Linear, ReLU, Softmax
from helpers import length_to_mask
from BaseNetwork import BaseNetwork
import torch

    
class GraphAttentionModel(BaseNetwork):  
    def __init__(self, 
                num_embeddings=7,
                embedding_dim=64,
                num_layers=1,
                num_heads=1,
                num_classes=5,
                name=None,
                edge_encoding=EDGE_ENCODING_TYPE.NONE):

        super(GraphAttentionModel, self).__init__(name=name)

        #TODO: edges embedding countains an embedding for the padding, which we later mask out
        if edge_encoding==EDGE_ENCODING_TYPE.ONEDIMENSIONAL:
            self.edges_embedding = Embedding(3, 1)
        elif edge_encoding==EDGE_ENCODING_TYPE.EMBEDDINGDIM:
            self.edges_embedding = Embedding(3, embedding_dim)
        else:
            self.edges_embedding = None
        self.embeddings = Embedding(num_embeddings, embedding_dim)
        self.softmax = Softmax(dim=1)   

        self.l_out = Linear(in_features = embedding_dim,out_features = num_classes)
    
        self.attention_layers = ModuleList(
            [GraphAttentionLayer(in_features=embedding_dim, out_features=embedding_dim, num_heads=num_heads, edge_encoding=edge_encoding) for _ in range(num_layers)]
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

        out['out'] = x = self.l_out(x[target_mask,:])
        
        out['prediction']=torch.argmax(self.softmax(x),dim=1)

        return out


if __name__ == "__main__":
    import argparse
    from dataloader import QM9AtomDataset
    from torchtext.data import Iterator
    from torch.optim import Adam
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_masks', default=1, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs',  default=100, type=int)
    parser.add_argument('--batch_size', default=248, type=int)
    parser.add_argument('--edge_encoding', default=1, type=int)
    parser.add_argument('--epsilon_greedy', default=0.2, type=float)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device=torch.device('cpu')

    data = QM9AtomDataset(data = 'data/adjacency_matrix_train.pkl',
                      include_mask_in_input=True,
                      num_masks=args.num_masks,
                      epsilon_greedy=args.epsilon_greedy)

    training, validation = data.split()

    train_iter = Iterator(
                        training,
                        batch_size =args.batch_size ,
                        device=device)
    val_iter = Iterator(
                        validation,
                        batch_size = args.batch_size,
                        device=device)

    graphAttentionModel = GraphAttentionModel(num_layers=args.num_layers,
                                            num_heads=args.num_heads, 
                                            embedding_dim=args.embedding_dim,
                                            edge_encoding=args.edge_encoding,
                                            name=(
                                                "GraphAttentionModel"
                                                f"_num_masks={args.num_masks}"
                                                f"_num_layers={args.num_layers}"
                                                f"_num_heads={args.num_heads}" 
                                                f"_embedding_dim={args.embedding_dim}"
                                                f"_lr={args.lr}"
                                                f"_edge_encoding={args.edge_encoding}"
                                                f"_epsilon_greedy={args.epsilon_greedy}"
                                            )
                        )

    optimizer_fun = lambda param:Adam(param, lr=args.lr)
    graphAttentionModel.train_network(train_iter, val_iter, num_epochs=args.num_epochs, eval_after_epochs=1, 
                  log_after_epochs=2, optimizer_fun=optimizer_fun, save_model=True)
    
    graphAttentionModel.confusion_matrix(class_labels=['H','C','O','N','F'])
    graphAttentionModel.plot_attention(next(iter(val_iter)))
    

    
