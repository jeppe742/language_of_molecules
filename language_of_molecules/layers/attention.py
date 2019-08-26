from torch.nn import Module, Linear, Dropout, LayerNorm, ReLU, Embedding
from torch.nn.functional import softmax, relu, leaky_relu
import numpy as np
import torch


class EDGE_ENCODING_TYPE:
    NONE=0
    ONEDIMENSIONAL=1
    EMBEDDINGDIM=2
    GRAPH=3
    RELATIVE_POSITION=4

class MultiHeadAttention(Module):
    def __init__(self, in_features, hidden_dim=None, out_features=None, num_heads=1, dropout=0.2, edge_encoding=EDGE_ENCODING_TYPE.NONE):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.edge_encoding = edge_encoding
        if out_features is None:
            out_features = in_features
        if hidden_dim is None:
            hidden_dim = in_features
            
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        
        self.dropout = Dropout(dropout)
       
    
        self.linear_query = Linear(in_features = in_features, out_features = num_heads*hidden_dim)
        self.linear_key = Linear(in_features = in_features, out_features = num_heads*hidden_dim)
        self.linear_value = Linear(in_features = in_features, out_features = num_heads*hidden_dim)
        
        self.linear_out = Linear(in_features = num_heads*hidden_dim, out_features = out_features)
        
    def forward(self, key, value, query, mask, adj=None, edges_embedding=None):
        '''
        Args
            key ([batch_size, seq_len, num_features])
            value ([batch_size, seq_len, num_features])
            query ([batch_size, seq_len, num_features])
            mask ([batch_size, seq_len])
        '''
        batch_size,seq_len,embedding_dim=value.shape
        
        #Project input
        key = self.linear_key(key)       #[batch_size, seq_len, num_heads*hidden_dim]
        value = self.linear_value(value) #[batch_size, seq_len, num_heads*hidden_dim]
        query = self.linear_query(query) #[batch_size, seq_len, num_heads*hidden_dim]
        
        #Expand heads to its own dimension
        key = key.view(batch_size, seq_len, self.num_heads, self.hidden_dim) #[batch_size, seq_len, num_heads, hidden_dim]
        value = value.view(batch_size, seq_len, self.num_heads, self.hidden_dim) #[batch_size, seq_len, num_heads, hidden_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.hidden_dim) #[batch_size, seq_len, num_heads, hidden_dim]
        
        #move num_heads dimension
        key = key.permute(0,2,3,1) #[batch_size, num_heads, hidden_dim, seq_len]
        value = value.permute(0,2,1,3) #[batch_size, num_heads, seq_len, hidden_dim]
        query = query.permute(0,2,1,3) #[batch_size, num_heads, seq_len, hidden_dim]
        
        attention = torch.matmul(query, key)/np.sqrt(self.hidden_dim) #[batch_size, num_heads,  seq_len, seq_len]
        
        if self.edge_encoding == EDGE_ENCODING_TYPE.RELATIVE_POSITION:
            a_k = edges_embedding[0](adj.long()) # [batch_size, seq_len, seq_len, hidden_dim]
            a_k = a_k.permute(0,1,3,2) #[batch_size, seq_len, hidden_dim, seq_len]
            query_tmp = query.permute(0,2,1,3) #[batch_size, seq_len, num_heads, hidden_dim]
            attention_k = torch.matmul(query_tmp, a_k)/np.sqrt(self.hidden_dim) #[batch_size, seq_len, num_heads, seq_len]
            attention_k = attention_k.permute(0,2,1,3) #[batch_size, num_heads, seq_len, seq_len]

            #Add the second term corresponding to the relative position representation
            attention += attention_k

        #make mask same dimensions as attention vector
        mask = mask.unsqueeze(2).unsqueeze(3).permute(0,3,2,1) #[batch_size, num_heads, seq_len, seq_len]

        #Mask out padding in attention
        attention.masked_fill_(mask, - float('inf'))


        if self.edge_encoding == EDGE_ENCODING_TYPE.ONEDIMENSIONAL:
            adj = adj.unsqueeze(3).permute(0,3,1,2).long()
            attention += edges_embedding(adj).squeeze(4)
        elif self.edge_encoding == EDGE_ENCODING_TYPE.GRAPH:
            adj = adj.unsqueeze(3).permute(0,3,1,2) #[batch_size, 1, seq_len, seq_len]
            #Mask out all but the connected atoms
            attention.masked_fill_(1-adj, -float('inf'))


        #Calculate normalized attention weights
        attention_weights = softmax(attention, dim = -1) #[batch_size, num_heads, seq_len, seq_len]
        if self.edge_encoding == EDGE_ENCODING_TYPE.GRAPH:
            #Some of the padded atoms have 0 neighbours, so their attentions is NaN, which we just replace with 0  
            attention_weights = torch.where(attention_weights!=attention_weights, torch.tensor(0., device=key.device), attention_weights)

        attention_weights = self.dropout(attention_weights)

        #Weighted sum of input
        context = torch.matmul(attention_weights, value) #[batch_size, num_heads, seq_len, hidden_dim]

        if self.edge_encoding == EDGE_ENCODING_TYPE.RELATIVE_POSITION:
            a_v = edges_embedding[1](adj.long()) #[batch_size, seq_len, seq_len, hidden_dim]
            attention_weights_tmp = attention_weights.permute(0,2,1,3) #[batch_Size, seq_len, num_heads, seq_len]
            context_v = torch.matmul(attention_weights_tmp, a_v) #[batch_size, seq_len, num_heads, hidden_dim]
            context_v = context_v.permute(0,2,1,3) #[batch_size, num_heads, seq_len, hidden_dim]
            
            context += context_v

        context = context.permute(0,2,1,3).reshape(batch_size, seq_len, self.num_heads*self.hidden_dim)
        
        out = self.linear_out(context)
        
        return out, attention_weights
    



class PositionwiseFeedForward(Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
    Args:
        in_features (int): the size of input for the first-layer of the FFN.
        hidden_dim (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, in_features, hidden_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Linear(in_features, hidden_dim)
        self.w_2 = Linear(hidden_dim, in_features)
        self.layer_norm = LayerNorm(in_features, eps=1e-6)
        self.dropout_1 = Dropout(dropout)
        self.relu = ReLU()
        self.dropout_2 = Dropout(dropout)

    def forward(self, x):
        """Layer definition.
        Args:
            x: ``(batch_size, input_len, in_features)``
        Returns:
            (FloatTensor): Output ``(batch_size, input_len, in_features)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x



class TransformerLayer(Module):
    def __init__(self, in_features, hidden_dim, num_heads=1, dropout=0.2, edge_encoding=EDGE_ENCODING_TYPE.NONE):
        super().__init__()

        self.attention = MultiHeadAttention(in_features=in_features, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, edge_encoding=edge_encoding)
        self.feed_forward = PositionwiseFeedForward(in_features = in_features, hidden_dim = hidden_dim, dropout=dropout)
        self.dropout = Dropout()
        self.layer_norm = LayerNorm(in_features)


    def forward(self, inputs, mask, adj=None, edges_embedding=None):

        
        inputs_normed = self.layer_norm(inputs)
        context, attn_weights = self.attention(inputs_normed, inputs_normed, inputs_normed, mask, adj=adj, edges_embedding=edges_embedding)
        out = self.dropout(context) + inputs
        return self.feed_forward(out), attn_weights


class GraphAttentionLayer(Module):
    def __init__(self, in_features, hidden_dim=None, out_features=None, num_heads=1, edge_encoding=EDGE_ENCODING_TYPE.NONE):
        super(GraphAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.edge_encoding = edge_encoding
        
        if out_features is None:
            out_features = in_features
        if hidden_dim is None:
            hidden_dim = in_features
            
        self.out_features = out_features
        self.hidden_dim = hidden_dim
       
    
        self.W = Linear(in_features = in_features, out_features = num_heads*hidden_dim)

       
        self.a = Linear(in_features = hidden_dim*2, out_features=1)

        self.linear_out = Linear(in_features = num_heads*hidden_dim, out_features = out_features)
        
    def forward(self, h, mask, adj=None, edge_embedding = None):
        '''
        Args
            h ([batch_size, seq_len, num_features])
            mask ([batch_size, seq_len])
            adj ([batch_size, seq_len, seq_len])
        '''
        batch_size,seq_len,embedding_dim=h.shape
        
        #Project input
        h = self.W(h)       #[batch_size, seq_len, num_heads*hidden_dim]

        #Expand num_heads and hidden_dim to two different dimensions
        h = h.unsqueeze(3).reshape(batch_size, seq_len, self.num_heads, self.hidden_dim)
        h = h.permute(0,2,1,3) #[batch_size, num_heads, seq_len, hidden_dim]

        #Create new tensor with h repeated seq_len times
        h_tmp = h.unsqueeze(4).expand(batch_size, self.num_heads,seq_len, self.hidden_dim, seq_len)
        h_tmp = h_tmp.permute(0,1,2,4,3) #[batch_size, num_heads, seq_len, seq_len , hidden_dim]
        
        if self.edge_encoding==EDGE_ENCODING_TYPE.EMBEDDINGDIM:
            adj = adj.unsqueeze(3).permute(0,3,1,2).long() #[batch_size, 1, seq_len, seq_len]
            h_tmp += edge_embedding(adj) #[batch_size, 1, seq_len, seq_len, hidden_dim]

        #Concat pairwise combinations of atoms in h_tmp
        a_input = torch.cat((h_tmp, h_tmp.permute(0,1,3,2,4)),dim=-1) #[batch_size, num_heads, seq_len, seq_len, 2*hidden_dim]



        #Linear layer over concatenations
        attention = leaky_relu(self.a(a_input).squeeze(4)) #[batch_size, num_heads, seq_len, seq_len]
        
        
   
        #make mask same dimensions as attention vector
        mask = mask.unsqueeze(2).unsqueeze(3).permute(0,3,2,1) #[batch_size, 1, 1, seq_len]
        #Mask out padding in attention
        attention.masked_fill_(mask, - float('inf'))


        if self.edge_encoding==EDGE_ENCODING_TYPE.ONEDIMENSIONAL:
            adj = adj.unsqueeze(3).permute(0,3,1,2).long()
            attention += edge_embedding(adj).squeeze(4)
        elif self.edge_encoding==EDGE_ENCODING_TYPE.GRAPH:
            adj = adj.unsqueeze(3).permute(0,3,1,2) #[batch_size, 1, seq_len, seq_len]
            #Mask out all but the connected atoms
            attention.masked_fill_(1-adj, -float('inf'))


        #Calculate normalized attention weights
        attention_weights = softmax(attention, dim = -1) #[batch_size, num_heads, seq_len, seq_len]
        
        if self.edge_encoding==EDGE_ENCODING_TYPE.GRAPH:
            #Some of the padded atoms have 0 neighbours, so their attentions is NaN, which we just replace with 0  
            attention_weights = torch.where(attention_weights!=attention_weights, torch.tensor(0., device=h.device), attention_weights)
        
        #Weighted sum of input
        context = torch.matmul(attention_weights, h) #[batch_size, num_heads, seq_len, hidden_dim]
    
        context = context.permute(0,2,1,3).reshape(batch_size, seq_len, self.num_heads*self.hidden_dim)
        
        out = relu(self.linear_out(context))
        
        return out, attention_weights
    
