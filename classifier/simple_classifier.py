import torch
import torch.nn as nn
import torch.nn.functional as F



"""
This class is meant to take the embeddings output from the SSL encoder
and output class classifications. It should work with the labeled dataset.
"""

class SimpleClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(SimpleClassifier, self).__init__()

        h1 = 1200
        h2 = 500
        h3 = 200

        #There's got to be a better way to write this...?
        layers = [ nn.Linear(1000,h1),
                   nn.Sigmoid(),
                   nn.Linear(h1,h2),
                   nn.Sigmoid(),
                   nn.Linear(h2,h3),
                   nn.Sigmoid(),
                   nn.Linear(h3,num_classes)
                 ]
        
        self.network = nn.ModuleList(layers)

    def forward(self, x):
        for f in self.network:
            x = f(x)
        return x

