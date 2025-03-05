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

    def forward(self, *, input: torch.tensor, hc_0: tuple[torch.tensor, torch.tensor]):
        #Performs matrix multiplication of previous hidden state and input
        h_0, c_0 = hc_0 #shape [hidden_size]
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
    def __init__(self, *, input_size: int, hidden_size: int, output_size: int):
        super().__init()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.bnlstmcell = BNLSTMCell(input_size=input_size,
                                     hidden_size=hidden_size)
        
        #Initiliase h_0 and c_0 - shape:  [N, H] -> [batch_size, hidden_size]
    def forward(self, input: torch.tensor, hc_0: tuple[torch.tensor, torch.tensor]= None):
        # input shape: [N, L, H_in] -> [batch_size, sequence length, input_size] - sequence length: time step

        # Get batch_size, sequence length and input_size

        # Initialise hc_0

        # Loop through each sample

        # Stack hidden state of each sample

        # return hidden_t, hc_0
        pass
        
class SPPArchitecture(torch.nn.Module):
    """Creates the architecture for Stock Price Prediction
    
    Args:
        input_size: number of elements in input vector
        hidden_size: number of elements in hidden units 
        output_shape: number of elements in output unit
    """
    def __init__(self, input_size: int , hidden_size: int, output_shape: int):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.bnlstm = BNLSTMCell(input_size=input_size,
                                 hidden_size=hidden_size)
        self.dropout = torch.nn.Dropout()
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=output_shape)
        
    def forward(self, input: torch.tensor):
        pass
