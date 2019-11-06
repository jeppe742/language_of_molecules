
import torch
from layers.BaseNetwork import BaseNetwork

class OctetRuleModel(BaseNetwork):
    def __init__(self,use_cuda=False):

        super(OctetRuleModel, self).__init__(name='OctetRule', use_cuda=use_cuda)

        self.bonds2atom = {1:0, 2:2, 3:3, 4:1} 
        self.bonds2prob = {1:torch.tensor([4.106302,0,0,0,-2.10630271]), 2:torch.tensor([0,0,1e9,0,0]), 3:torch.tensor([0,0,0,1e9,0]), 4:torch.tensor([0,1e9,0,0,0])}

    def forward(self, batch):
        out = {}

        atoms = batch.atoms_num
        batch_size, num_atoms = atoms.shape
        lengths = batch.lengths

        adj = batch.adj

        num_neighbours = torch.sum(adj, dim=-1)
        
        x = torch.zeros((batch_size,num_atoms, 5)).to(atoms.device)
        for b in range(batch_size):
            for a in range(lengths[b]):
                x[b,a,:] = self.bonds2prob[num_neighbours[b,a].item()]
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