
import torch
from layers.BaseNetwork import BaseNetwork
from torch.nn import Embedding
import sys
sys.path.append('~/dtu/language_of_molecules')
from utils.helpers import inverse_softmax

class OctetRuleModel(BaseNetwork):
    def __init__(self,use_cuda=False, k=1842, dataset='zinc'):

        super(OctetRuleModel, self).__init__(name='OctetRule', use_cuda=use_cuda)

        #Unigram probabilities

        self.probs_zinc = torch.tensor([
            [0      , 0      , 0     , 0     , 0    , 0 , 0    , 0    , 0    , 0  ],
            [3537247, 0      , 0     , 0     , 63837, 0 , 0    , 33729, 10744, 802],
            [0      , 0      , 404125, 0     , 0    , 0 , 68119, 0    ,  0   , 0  ],
            [0      , 0      , 0     , 455800, 0    , 90, 0    , 0    ,  0   , 0  ],
            [0      , 2886876, 0     , 0     , 0    , 0 , 0    , 0    ,  0   , 0  ],
            [0      , 0      , 0     , 0     , 0    , 0 , 0    , 0    , 0    , 0  ],
            [0      , 0      , 0     , 0     , 0    , 0 , 0    , 0    , 0    , 0  ]
        ],dtype=torch.float64)

        self.probs_qm9 = torch.tensor([
            [     1,      1,      1,      1,      1],
            [873518,      0,      0,      0,   3037],
            [     0,      0, 130432,      0,      0],
            [     0,      0,      0,  91173,      0],
            [     0, 584330,      0,      0,      0]
        ],dtype=torch.float64)

        if dataset=='zinc':
            # k smoothing
            self.probs_zinc += k
            #normalize counts
            self.probs = self.probs_zinc/self.probs_zinc.sum(dim=1,keepdim=True)

            self.bonds2prob = Embedding(7,10)
        else:
            self.probs = self.probs_qm9/self.probs_qm9.sum(dim=1,keepdim=True)

            self.bonds2prob = Embedding(5,5)

        self.bonds2prob.weight.data.copy_(
            inverse_softmax(self.probs)
        )
    def forward(self, batch):
        out = {}

        num_neighbours = batch.num_neighbours
        
        x = self.bonds2prob(num_neighbours)

        out['out'] = x

        out['prediction'] = torch.argmax(x, dim=-1)

        return out
