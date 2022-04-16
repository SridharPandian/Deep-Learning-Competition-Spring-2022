import torch
import torch.nn as nn
import torch.nn.functional as F



"""
This class is meant to take the embeddings output from the SSL encoder
and output class classifications. It should work with the labeled dataset.
"""

class SimpleClassifier(nn.Module):
def __init__(self):
    super(SimpleClassifier, self).__init__()


def forward(self, x):

