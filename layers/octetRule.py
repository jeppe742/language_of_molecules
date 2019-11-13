
import torch
from layers.BaseNetwork import BaseNetwork
from torch.nn import Embedding
import sys
sys.path.append('~/dtu/language_of_molecules')
from utils.helpers import inverse_softmax

class OctetRuleModel(BaseNetwork):
    def __init__(self,use_cuda=False, k=1842):

        super(OctetRuleModel, self).__init__(name='OctetRule', use_cuda=use_cuda)

        #Unigram probabilities
        self.probs = torch.tensor([
            [0      , 0      , 0     , 0     , 0    , 0 , 0    , 0    , 0    , 0  ],
            [3537247, 0      , 0     , 0     , 63837, 0 , 0    , 33729, 10744, 802],
            [0      , 0      , 404125, 0     , 0    , 0 , 68119, 0    ,  0   , 0  ],
            [0      , 0      , 0     , 455800, 0    , 90, 0    , 0    ,  0   , 0  ],
            [0      , 2886876, 0     , 0     , 0    , 0 , 0    , 0    ,  0   , 0  ],
            [0      , 0      , 0     , 0     , 0    , 0 , 0    , 0    , 0    , 0  ],
            [0      , 0      , 0     , 0     , 0    , 0 , 0    , 0    , 0    , 0  ]
            ],dtype=torch.float64)
        # k smoothing
        self.probs += k
        #normalize counts
        self.probs = self.probs/self.probs.sum(dim=1,keepdim=True)

        self.bonds2prob = Embedding(7,10)


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
