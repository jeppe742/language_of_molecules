
import torch
from layers.BaseNetwork import BaseNetwork
from torch.nn import Embedding

class OctetRuleModel(BaseNetwork):
    def __init__(self,use_cuda=False):

        super(OctetRuleModel, self).__init__(name='OctetRule', use_cuda=use_cuda)
        self.bonds2prob = Embedding(7,10)
        self.bonds2prob.weight.data.copy_(
            torch.tensor(
            [
                [0         , 0   , 0          , 0          , 0          , 0          , 0          , 0          , 0         , 0],
                [15.62359591,  0.54473633,  0.54473633,  0.54473633, 11.60884023,  0.54473633,  0.54473633, 10.97087926,  9.82693214,  7.23309104],
                [-0.51536451, -0.51536451, 12.39411748, -0.51536451, -0.51536451, -0.51536451, 10.61366163, -0.51536451, -0.51536451, -0.51536451],
                [-0.8577243,  -0.8577243,  -0.8577243,  12.17208727, -0.8577243,   3.65313519, -0.8577243,  -0.8577243,  -0.8577243,  -0.8577243 ],
                [-2.10050063, 12.77512012, -2.10050063, -2.10050063, -2.10050063, -2.10050063, -2.10050063, -2.10050063, -2.10050063, -2.10050063],
                [0         , 0   , 0          , 0          , 0          , 0          , 0          , 0          , 0         , 0],
                [0         , 0   , 0          , 0          , 0          , 0          , 0          , 0          , 0         , 0]
                # [0         , 0   , 0          , 0          , 0          , 0          , 0          , 0          , 0         , 0],
                # [5.86486708, -1e9, -1e9       , -1e9       ,  1.84759282, -1e9       , -1e9       , 1.20891555 , 0.06240334, -2.53122385],
                # [-1e9      , -1e9, 21.34628232, -1e9       , -1e9       , -1e9       , 19.56511109, -1e9       , -1e9      , -1e9],
                # [-1e9      , -1e9, -1e9       , 20.87770471, -1e9       , 12.34883891, -1e9       , -1e9       , -1e9      , -1e9],
                # [0         , 1e9 , 0          , 0          , 0          , 0          , 0          , 0          , 0         , 0],
                # [0         , 0   , 0          , 0          , 0          , 0          , 0          , 0          , 0         , 0],
                # [0         , 0   , 0          , 0          , 0          , 0          , 0          , 0          , 0         , 0]
                ]))

            # torch.tensor(
            # [
            #     [0       , 0   , 0   , 0  , 0 ],
            #     [4.106302, -1e9, -1e9,-1e9, -2.10630271],
            #     [0       , 0   , 1e9 , 0  , 0],
            #     [0       , 0   , 0   , 1e9, 0 ],
            #     [0       , 1e9 , 0   , 0  , 0]
            #     ]))
        # self.bonds2atom = {1:0, 2:2, 3:3, 4:1} 
        # self.bonds2prob = {1:torch.tensor([4.106302,0,0,0,-2.10630271]), 2:torch.tensor([0,0,1e9,0,0]), 3:torch.tensor([0,0,0,1e9,0]), 4:torch.tensor([0,1e9,0,0,0])}

#ATOMS = ['H','C','O','N','F','P','S','Cl','Br','I','M']
        # self.bonds2atom = {1:0, 2:2, 3:3, 4:1} 
        # self.bonds2prob = {
        #     1:torch.tensor([5.86486708, -1e9, -1e9       , -1e9,  1.84759282, -1e9, -1e9       , 1.20891555 , 0.06240334, -2.53122385]), 
        #     2:torch.tensor([-1e9      , -1e9, 21.34628232, -1e9, -1e9       , -1e9, 19.56511109, -1e9       , -1e9      , -1e9]), 
        #     3:torch.tensor([-1e9      , -1e9, -1e9       , -1e9, 20.87770471, -1e9, -1e9       , 12.34883891, -1e9      , -1e9]), 
        #     4:torch.tensor([0         , 1e9 , 0          , 0   , 0          , 0   , 0          , 0          , 0         , 0]),
        #     5:torch.tensor([0         , 0   , 0          , 0   , 0          , 0   , 0          , 0          , 0         , 0]),
        #     6:torch.tensor([0         , 0   , 0          , 0   , 0          , 0   , 0          , 0          , 0         , 0]) }

    def forward(self, batch):
        out = {}

        atoms = batch.atoms_num
        batch_size, num_atoms = atoms.shape
        lengths = batch.lengths

        adj = batch.adj

        num_neighbours = torch.sum(adj, dim=-1)
        
        #x = torch.zeros((batch_size,num_atoms, 10)).to(atoms.device)
        x = self.bonds2prob(num_neighbours)
        # for b in range(batch_size):
        #     for a in range(lengths[b]):
        #         x[b,a,:] = self.bonds2prob[num_neighbours[b,a].item()]
                #x[b,a, self.bonds2atom[num_neighbours[b,a].item()]] = 1
        #Multiply with large number, so ensure softmax(x) has only 1 non-zero entry
        out['out'] = x

        out['prediction'] = torch.argmax(x, dim=-1)

        return out


class OctetRule():
    def __init__(self):
        self.bonds2atoms = {1:'H,F', 2:'O', 3:'N', 4:'C'} 

    def __call__(self, batch):

        atoms = batch.atoms_num
        batch_size, num_atoms = atoms.shape


        adj = batch.adj
        target_mask = batch.target_mask

        num_neighbours = torch.sum(adj, dim=-1)
        
        predictions = []
        #x = torch.zeros((batch_size,num_atoms, 5)).to(atoms.device)
        for b in range(batch_size):
            tmp = []
            for num_neighbour in num_neighbours[b, target_mask[b,:]]:
                tmp += [self.bonds2atoms[num_neighbour.item()]]
            predictions += [tmp]

        return predictions