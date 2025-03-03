import pandas as pd 
import torch
import os 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SMP_Dataset(Dataset):
    """A class inherited from torch.utils.data.Dataset for loading CSV file for Stock Market Prediction"""
    def __init__(self, path: os.path, target_col: str):
        self.data = pd.read_csv(path)
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data.drop(self.target_col, axis = 1).select_dtypes('number')
        target = self.data[self.target_col]
        return torch.tensor(features.iloc[index,:].values), torch.tensor(target[index])
    
def create_dataloaders(
        train_path: os.path,
        test_path: os.path,
        target_col: str,
        batch_size: int
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
    A tuple (train_dataloaders, test_dataloaders)."""

    #Initialise data
    train_data = SMP_Dataset(train_path, target_col)
    test_data = SMP_Dataset(test_path, target_col)

    #Create training and testing DataLoaders
    train_dataloaders = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True)
    
    test_dataloaders = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True
    )

    return train_dataloaders, test_dataloaders


    
