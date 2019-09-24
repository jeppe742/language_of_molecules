
import torch
from layers.BaseNetwork import BaseNetwork

class OctetRuleModel(BaseNetwork):
    def __init__(self,use_cuda=False):

        super(OctetRuleModel, self).__init__(name='OctetRule', use_cuda=use_cuda)

        self.bonds2atom = {1:0, 2:2, 3:3, 4:1} 

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
                x[b,a, self.bonds2atom[num_neighbours[b,a].item()]] = 1
        #Multiply with large number, so ensure softmax(x) has only 1 non-zero entry
        out['out'] = x*1e9

        out['prediction'] = torch.argmax(x, dim=-1)

        return out