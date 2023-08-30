import torch
import torch.nn as nn


class LSTMmodel(nn.Module):
  def __init__(self, hidden_size, input_size, nbr_layers, nbr_classes, dropouts, device):
    super(LSTMmodel, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.nbr_layers = nbr_layers
    self.device = device

    self.lstm = nn.LSTM(input_size=self.input_size, 
                        hidden_size=self.hidden_size, 
                        num_layers=self.nbr_layers, 
                        batch_first=True,
                        dropout = dropouts)
    # after the batch first the shape of input and output should be (batch, sequence==time, features)
    self.fc = nn.Linear(self.hidden_size, nbr_classes)
  
  def forward(self, x):
    # x.shape[0] == batch_size
    hidden_state = torch.zeros(self.nbr_layers, x.shape[0], self.hidden_size).to(self.device)
    cell_state = torch.zeros(self.nbr_layers, x.shape[0], self.hidden_size).to(self.device)

    out, _ = self.lstm(x, (hidden_state, cell_state))
    out = self.fc(out[:, -1, :])

    return out
