from layers.BaseNetwork import BaseNetwork
import torch
import numpy as np
import sys
sys.path.append('~/dtu/language_of_molecules')
from utils.helpers import inverse_softmax

class UnigramModel(BaseNetwork):  
    def __init__(self,
                name=None, dataset='qm9'):

        super(UnigramModel, self).__init__(name=name)

        if dataset=='qm9':
            self.prob = torch.tensor([0.519182,0.347301,0.077523,0.054189,0.001805])
        else:
            self.prob = torch.tensor([0.474075, 0.386910, 0.054162, 0.061088, 0.008556, 0.000012, 0.009130, 0.004520, 0.001440, 0.000107])
        self.prob = inverse_softmax(self.prob.reshape(1,-1))
    def forward(self, batch):
        out={}

        targets = batch.targets_num
        atoms = batch.atoms_num
       
        #These are the atomic densities from the training dataset
        #[0.51148007,0.35092365,0.07813921,0.05803061,0.00142646]
        # Since the rest of our models output unormalized distributions, we solved for the inverse softmax values using least-squares optimization
        # out['out'] = torch.tensor([2.5536360,  2.1769042,  0.6748235,  0.3772931, -3.3294513], device=targets.device).expand(atoms.size(0),atoms.size(1),-1)
        out['out'] = self.prob.expand(atoms.size(0),atoms.size(1),-1)
        
        out['prediction']=torch.argmax(out['out'],dim=-1)
 
        return out
