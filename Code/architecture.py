import torch

#Reference:
# 1. https://github.com/hellozgy/bnlstm-pytorch
# 2. https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# 3. https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BNLSTMCell(torch.nn.Module):
    """A BNLSTM cell used for internal calculation."""
    def __init__(self, *, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for 4 gates: input, forget, cell, output
        self.weight_hidden = torch.nn.Parameter(torch.randn(hidden_size, 4*hidden_size).to(device))
        self.weight_input = torch.nn.Parameter(torch.randn(input_size, 4*hidden_size).to(device))

        # Initialise bias
        self.bias = torch.nn.Parameter(torch.zeros(4*hidden_size).to(device))

        # Batch normalization layers
        self.bn_hidden = torch.nn.BatchNorm1d(4*hidden_size).to(device)
        self.bn_input = torch.nn.BatchNorm1d(4*hidden_size).to(device)
        self.bn_c = torch.nn.BatchNorm1d(hidden_size).to(device)

    def forward(self, input: torch.tensor, hc_0: tuple[torch.tensor, torch.tensor]):
        """Performs forward pass 
        
        Args:
            input: input vector which has shape (batch_size, time_step)
            hc_0: a tuple of the form (previous_hidden_state, previous_cell)
            
        Returns:
            A tuple which contains new hidden state and new cell
            In the form (h_1, c_1)"""
        #Performs matrix multiplication of previous hidden state and input
        h_0, c_0 = hc_0 #shape [batch_size, hidden_size]
        
        hidden_trans = torch.matmul(h_0, self.weight_hidden)
        input_trans = torch.matmul(input, self.weight_input)

        # Calculate and apply batch normalization to four gates
        four_gates = self.bn_hidden(hidden_trans) + self.bn_input(input_trans) + self.bias

        # Split to get specific four gates
        f, i, o, g = torch.chunk(four_gates, chunks=4, dim=1)

        # Compute new hidden state
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1))

        return h_1, c_1
    
class BNLSTM(torch.nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.bnlstmcell = BNLSTMCell(input_size=input_size,
                                     hidden_size=hidden_size)
        
        #Initiliase h_0 and c_0 - shape:  [N, H] -> [batch_size, hidden_size]
        self.h_0 = torch.nn.Parameter(torch.zeros(1, hidden_size).to(device))
        self.c_0 = torch.nn.Parameter(torch.zeros(1, hidden_size).to(device))

    def forward(self, input: torch.tensor, hc_0: tuple[torch.tensor, torch.tensor]= None):
        input = input.to(device)
        batch_size, input_size = input.size() # input shape: [N, L] -> [batch_size, input_size]
        if hc_0 is None: # Initialise hc_0
            hc_0 = (self.h_0.repeat(batch_size, 1), self.c_0.repeat(batch_size, 1))
        h_1, c_1 = self.bnlstmcell(input=input,
                                   hc_0=hc_0)
        hc_1 = (h_1, c_1)

        return h_1, hc_1
    
class SPP(torch.nn.Module):
    """Creates the architecture for Stock Price Prediction
    
    Args:
        input_size: number of elements in input vector
        hidden_size: number of elements in hidden units 
        output_shape: number of elements in output unit
    """
    def __init__(self, input_size: int , hidden_size: int, output_shape: int):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.bnlstm = BNLSTM(input_size=input_size,
                             hidden_size=hidden_size)
        self.dropout = torch.nn.Dropout()
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=output_shape)
        
    def forward(self, x: torch.tensor):
        x = self.relu(x)
        output_bnlstm, hc = self.bnlstm(x)
        output_dropout = self.dropout(output_bnlstm)
        output_linear = self.linear(output_dropout)
        return output_linear
