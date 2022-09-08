import torch
import torch.nn as nn

class Network(nn.Module):
  
  def __init__(self):

    super().__init__()

    self.hidden = nn.Linear(13, 12)
    self.relu = nn.ReLU()

    self.output = nn.Linear(12, 3)  
    self.softMax = nn.LogSoftmax(dim=-1)
    return


  def forward(self, x):

    x = self.hidden(x)
    x = self.relu(x)

    x = self.output(x)
    x = self.softMax(x)
    return x