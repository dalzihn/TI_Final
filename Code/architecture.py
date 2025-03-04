import torch

class BNLSTMCell(torch.nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for 4 gates: input, forget, cell, output
        self.weight_hidden = torch.nn.Parameter(torch.randn(hidden_size, 4*hidden_size))
        self.weight_input = torch.nn.Parameter(torch.randn(input_size, 4*hidden_size))

        # Initialise bias
        self.bias = torch.nn.Parameter(torch.zeros(4*hidden_size))

        # Batch normalization layers
        self.bn_hidden = torch.nn.BatchNorm1d(4*hidden_size)
        self.bn_input = torch.nn.BatchNorm1d(4*hidden_size) 
        self.bn_c = torch.nn.BatchNorm1d(hidden_size)

    def forward(self, *, input: torch.tensor, hidden_prev: torch.tensor, cell_prev: torch.tensor):
        #Performs matrix multiplication of previous hidden state and input
        hiddenprev_trans = torch.matmul(hidden_prev, self.weight_hidden)
        input_trans = torch.matmul(input, self.weight_input)

        # Calculate and apply batch normalization to four gates
        four_gates = self.bn_hidden(hiddenprev_trans) + self.bn_input(input_trans) + self.bias

        # Split to get specific four gates
        f, i, o, g = torch.chunk(four_gates, chunks=4, dim=1)

        # Compute new hidden state
        new_cell = torch.sigmoid(f) * cell_prev + torch.sigmoid(i) * torch.tanh(g)
        new_hidden = torch.sigmoid(o) * torch.tanh(self.bn_c(new_cell))

        return new_hidden, new_cell
        
class BNLSTM(torch.nn.Module):
    pass
