import torch
import math
#Reference:
# 1. https://github.com/hellozgy/bnlstm-pytorch
# 2. https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# 3. https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 
class TransformerGRU_model(torch.nn.Module):
    """Creates the Transformer-based BNLSTM for Stock Price Prediction
    
    Args:
        input_size: number of elements in input vector
        hidden_size: number of elements in hidden units 
        output_shape: number of elements in output unit
    """
    def __init__(self, input_size: int , hidden_size: int, output_shape: int):
        super().__init__()       
         # NOTE: LSTM
        # First layer
        self.relu1 = torch.nn.ReLU()
        self.gru1 = torch.nn.GRU(input_size=input_size,
                                   hidden_size=hidden_size, 
                                   batch_first=True,
                                   num_layers=2)
        self.dropout1 = torch.nn.Dropout()

        # NOTE: Transformer addded
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                                          nhead=16,
                                                                          batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                               num_layers=2)
        
        # Second layer
        # self.relu2 = torch.nn.ReLU()
        self.gru2 = torch.nn.GRU(input_size=hidden_size,
                                   hidden_size=hidden_size, 
                                   batch_first=True,
                                   num_layers=2)
        self.dropout2 = torch.nn.Dropout()
       
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=output_shape)
        
    def forward(self, x: torch.tensor):
        # NOTE: LSTM
        # First layer
        x = self.relu1(x)
        x, h1 = self.gru1(x)
        x = self.dropout1(x)

        # NOTE: Transfomer added
        x = self.transformer_encoder(x)

        # Second layer
        # x = self.relu2(x)
        x, h2 = self.gru2(x, h1)
        x = self.dropout2(x)
        x = x[:, -1, :]
        output = self.linear(x)
        return output
    
class LSTM_model(torch.nn.Module):
    """Creates the LSTM for Stock Price Prediction
    
    Args:
        input_size: number of elements in input vector
        hidden_size: number of elements in hidden units 
        output_shape: number of elements in output unit
    """
    def __init__(self, input_size: int , hidden_size: int, output_shape: int):
        super().__init__()
        # NOTE: LSTM
        # First layer
        self.relu1 = torch.nn.ReLU()
        self.lstm1 = torch.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size, 
                                   batch_first=True,
                                   num_layers=2)
        self.dropout1 = torch.nn.Dropout()
        
        # Second layer
        self.relu2 = torch.nn.ReLU()
        self.lstm2 = torch.nn.LSTM(input_size=hidden_size,
                                   hidden_size=hidden_size, 
                                   batch_first=True,
                                   num_layers=2)
        self.dropout2 = torch.nn.Dropout()

        #   # Third layer
        # self.relu3 = torch.nn.ReLU()
        # self.lstm3 = torch.nn.LSTM(input_size=hidden_size,
        #                            hidden_size=hidden_size, 
        #                            batch_first=True,
        #                            num_layers=2)
        # self.dropout3 = torch.nn.Dropout()

        #  # Fourth layer
        # self.relu4 = torch.nn.ReLU()
        # self.lstm4 = torch.nn.LSTM(input_size=hidden_size,
        #                            hidden_size=hidden_size, 
        #                            batch_first=True,
        #                            num_layers=2)
        # self.dropout4 = torch.nn.Dropout()

        #  # Fifth layer
        # self.relu5 = torch.nn.ReLU()
        # self.lstm5 = torch.nn.LSTM(input_size=hidden_size,
        #                            hidden_size=hidden_size, 
        #                            batch_first=True,
        #                            num_layers=2)
        # self.dropout5 = torch.nn.Dropout()


        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=output_shape)

    def forward(self, x: torch.tensor):
        # NOTE: LSTM
        # First layer
        x = self.relu1(x)
        x, hc1 = self.lstm1(x)
        x = self.dropout1(x)

        # Second layer
        x = self.relu2(x)
        x, hc2 = self.lstm2(x, hc1)
        x = self.dropout2(x)

        #  # Third layer
        # x = self.relu3(x)
        # x, hc3 = self.lstm2(x, hc2)
        # x = self.dropout3(x)

        #  # Fourth layer
        # x = self.relu4(x)
        # x, hc4 = self.lstm4(x, hc3)
        # x = self.dropout4(x)

        #  # Fifth layer
        # x = self.relu5(x)
        # x, hc5 = self.lstm5(x, hc4)
        # x = self.dropout5(x)

        x = x[:, -1, :]
        output = self.linear(x)
        return output
    
class TransformerLSTM_model(torch.nn.Module):
    """Creates the Transformer-based BNLSTM for Stock Price Prediction
    
    Args:
        input_size: number of elements in input vector
        hidden_size: number of elements in hidden units 
        output_shape: number of elements in output unit
    """
    def __init__(self, input_size: int , hidden_size: int, output_shape: int):
        super().__init__()       
        # # NOTE: transfomer added
        # self.pos_encoding = PositionalEncoding(d_model=input_size,
        #                                        max_len=hidden_size)
        # Transfomer encoder layer
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                                          nhead=16,
                                                                          batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                               num_layers=2)
         # NOTE: LSTM
        # First layer
        self.relu1 = torch.nn.ReLU()
        self.lstm1 = torch.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size, 
                                   batch_first=True,
                                   num_layers=2)
        self.dropout1 = torch.nn.Dropout()
        
        # Second layer
        self.relu2 = torch.nn.ReLU()
        self.lstm2 = torch.nn.LSTM(input_size=hidden_size,
                                   hidden_size=hidden_size, 
                                   batch_first=True,
                                   num_layers=2)
        self.dropout2 = torch.nn.Dropout()
       
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=output_shape)
        
    def forward(self, x: torch.tensor):
        # NOTE: Transfomer added
        # x = self.pos_encoding(x)
        # x = self.transformer_encoder(x)

        # NOTE: LSTM
        # First layer
        x = self.relu1(x)
        x, hc1 = self.lstm1(x)
        x = self.dropout1(x)

        x = self.transformer_encoder(x)

        # Second layer
        x = self.relu2(x)
        x, hc2 = self.lstm2(x, hc1)
        x = self.dropout2(x)
        x = x[:, -1, :]
        output = self.linear(x)
        return output

