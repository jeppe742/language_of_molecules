from layers.BaseNetwork import BaseNetwork
import torch
import numpy as np


class UnigramModel(BaseNetwork):  
    def __init__(self,val_iter,
                name=None):

        super(UnigramModel, self).__init__(name=name)

        self.val_iter=val_iter

    def forward(self, batch):
        out={}

        batch_size, molecule_size = atoms.shape

        targets = batch.target

        targets = targets[targets != 0]
        targets -= 1 

        #These are the atomic densities from the training dataset
        #[0.51148007,0.35092365,0.07813921,0.05803061,0.00142646]
        # Since the rest of our models output unormalized distributions, we solved for the inverse softmax values using least-squares optimization
        out['out'] = torch.tensor([2.5536360,  2.1769042,  0.6748235,  0.3772931, -3.3294513], device=targets.device).expand(len(targets),-1)
        
        out['prediction']=torch.argmax(out['out'],dim=1)
 
        return out
