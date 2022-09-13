import torch
import torch.nn as nn

class Network(nn.Module):
  def __init__(self):
    super().__init__()

    self.hidden = nn.Linear(14, 64)
    self.relu = nn.ReLU()
    self.hidden2 = nn.Linear(64, 64)
    self.relu2 = nn.ReLU()
    self.output = nn.Linear(64, 2)  
    self.softmax = nn.LogSoftmax(dim=-1)


  def forward(self, x):
    x = self.hidden(x)
    x = self.relu(x)
    x = self.hidden2(x)
    x = self.relu2(x)
    x = self.output(x)
    x = self.softmax(x)
    return x
