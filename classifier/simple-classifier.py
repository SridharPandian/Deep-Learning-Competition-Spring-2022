import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP


"""
This class is meant to take the embeddings output from the SSL encoder
and output class classifications. It should work with the labeled dataset.
"""

class SimpleClassifier(nn.Module):
    def __init__(self, ss_model, n_classes, hidden_classifier_sizes = [512,256]):
        super(SimpleClassifier, self).__init__()
        self.backbone = ss_model #Self supervised trained model
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        #Output shape of backbone =1000
        self.classifier = nn.Sequential(
                    MLP(d_in =1000, d_out= n_classes , hidden_sizes = hidden_classifier_sizes),
                    nn.Softmax()
                    )
        #TODO - Use torch summary to check if the weights are frozen
        self.model = nn.Sequential(self.backbone,self.classifier)

    def forward(self, x):
        return self.model(x)

