import pandas as pd 
import torch
import os 
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses data
    
    Arg:
        data: a pandas DataFrame
    
    Returns:
        A pandas DataFrame that are preprocessed"""
    preprocessed_data = data
    # Drop the first two rows
    preprocessed_data = preprocessed_data.iloc[2:].reset_index(drop=True)

    # Change "Attributes" to "Data"
    preprocessed_data = preprocessed_data.rename({"Attributes" : "Date"}, 
                                             axis=1)
    
    # Chaneg data type
    convert_dict = {"high": float, "low": float, "open" : float, "close": float, 
                    "adjust": float, "volume_match": float, "value_match": float}
    preprocessed_data = preprocessed_data.astype(convert_dict)

    # TODO: Encode datetime
    preprocessed_data["Date"] = pd.to_datetime(preprocessed_data["Date"])
    # sine and cosine transformation of year
    preprocessed_data["year_sin"] = np.sin((2 * np.pi * preprocessed_data["Date"].dt.year) / 365)
    preprocessed_data["year_cos"] = np.cos((2 * np.pi * preprocessed_data["Date"].dt.year) / 365)

    # sine and cosine transformation of month
    preprocessed_data["month_sin"] = np.sin((2 * np.pi * preprocessed_data["Date"].dt.month) / 12)
    preprocessed_data["month_cos"] = np.cos((2 * np.pi * preprocessed_data["Date"].dt.month) / 12)

    # sine and cosine transformation of day
    preprocessed_data["day_sin"] = np.sin((2 * np.pi * preprocessed_data["Date"].dt.day) / 7)
    preprocessed_data["day_cos"] = np.cos((2 * np.pi * preprocessed_data["Date"].dt.day) / 7)

    # Drop date and code column
    preprocessed_data.drop(["Date", "code"], axis = 1, inplace = True)
    return preprocessed_data
class SPP_Dataset(Dataset):
    """A class inherited from torch.utils.data.Dataset for loading CSV file for Stock Price Prediction"""
    def __init__(self, path: os.path, target_col: str):
        self.data = preprocess(pd.read_csv(path))
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data.drop(self.target_col, axis = 1).select_dtypes('number')
        target = self.data[self.target_col]
        return torch.tensor(features.iloc[index,:].values, dtype=torch.float32), torch.tensor(target[index], dtype=torch.float32)
    
def create_dataloaders(
        train_path: os.path = None,
        test_path: os.path = None,
        target_col: str = None,
        batch_size: int = 64
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Creates training and testing DataLoaders

    Takes two paths which are training path and testing path respectively, first turns them into Pytorch Dataset
    and then into Pytorch DatasetLoaders

    Args:
        train_path: path that links to train file 
        test_path: path that links to test file
        target_col: the target column
        batch_size: number of samples in a batch
    
    Returns:
        If one of the paths is None, the function will return Pytorch Dataloader of the path which is not.
        Otherwise, returns both."""

    #Initialise dat
    train_dataloaders = None
    test_dataloaders = None
    if train_path: 
        train_data = SPP_Dataset(train_path, target_col)
        train_dataloaders = DataLoader(train_data, 
                                       batch_size=batch_size, 
                                       shuffle=True)
        
    if test_path: 
        test_data = SPP_Dataset(test_path, target_col)
        test_dataloaders = DataLoader(test_data,
                                      batch_size=batch_size,
                                      shuffle=True)
    

    if train_dataloaders and test_dataloaders:
        return train_dataloaders, test_dataloaders
    elif train_dataloaders:
        return train_dataloaders
    else:
        return test_dataloaders 

    
