import pandas as pd 
import torch
import os 
import numpy as np
import zipfile
import shutil
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses data
    
    Arg:
        data: a pandas DataFrame
    
    Returns:
        A pandas DataFrame that are preprocessed"""
    newdf = data
    newdf = newdf.dropna()
    scaler = MinMaxScaler()
    # Drop the first two rows
    newdf = newdf.iloc[2:].reset_index(drop=True)
    newdf = newdf.sort_values(by="Attributes", ignore_index=True)

    # Change "Attributes" to "Data"
    # newdf = newdf.rename({"Attributes" : "Date"}, 
    #                                          axis=1)
    newdf.drop("Attributes", axis = 1, inplace = True)

    # Chaneg data type
    convert_dict = {"high": float, "low": float, "open" : float, "close": float, 
                    "adjust": float, "volume_match": float, "value_match": float}
    newdf = newdf.astype(convert_dict)

    #Scale value
    newdf[['high', 'low', 'open', 'adjust', 'volume_match', 'value_match']] = scaler.fit_transform(newdf[['high', 'low', 'open', 'adjust', 'volume_match', 'value_match']])
    newdf[['high', 'low', 'open', 'adjust', 'volume_match', 'value_match']] = round(newdf[['high', 'low', 'open', 'adjust', 'volume_match', 'value_match']], 6)
    
    return newdf

class SPP_Dataset(Dataset):
    """A class inherited from torch.utils.data.Dataset for loading CSV file for Stock Price Prediction"""
    def __init__(self, path: os.path, target_col: str, timestep: int = 1):
        self.data = pd.read_csv(path, index_col=0)
        # self.data = preprocess(pd.read_csv(path))
        self.target_col = target_col
        self.timestep = timestep

    def __len__(self):
        # return len(self.data)
        return len(self.data) - self.timestep

    def __getitem__(self, index):
        features = self.data.drop(self.target_col, axis = 1).select_dtypes('number')
        target = self.data[self.target_col]

        # NOTE: added time step
        feature_seq = features.iloc[index: index + self.timestep].values
        target_value = target.iloc[index + self.timestep]
        return torch.tensor(feature_seq, dtype=torch.float32), torch.tensor(target_value, dtype=torch.float32)
        # return torch.tensor(target.iloc[index:index+self.timestep].values, dtype=torch.float32), torch.tensor(target.iloc[index + self.timestep], dtype=torch.float32)
    
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

    # #Initialise dat
    # train_dataloaders = None
    # test_dataloaders = None
    # val_dataloaders = None
    # if train_path: 
    #     train_data = SPP_Dataset(train_path, target_col)
    #     train_dataloaders = DataLoader(train_data, 
    #                                    batch_size=batch_size, 
    #                                    shuffle=True)
    
    # if val_path: 
    #     val_data = SPP_Dataset(val_path, target_col)
    #     val_dataloaders = DataLoader(val_data, 
    #                                  batch_size=batch_size, 
    #                                  shuffle=True)
        
    # if test_path: 
    #     test_data = SPP_Dataset(test_path, target_col)
    #     test_dataloaders = DataLoader(test_data,
    #                                   batch_size=batch_size,
    #                                   shuffle=True)
    
    # if train_dataloaders and test_dataloaders and val_dataloaders:
    #     return train_dataloaders, val_dataloaders, test_dataloaders
    # elif train_dataloaders and test_dataloaders:
    #     return train_dataloaders, test_dataloaders
    # elif train_dataloaders and val_dataloaders:
    #     return train_dataloaders, val_dataloaders
    # elif test_dataloaders and val_dataloaders:
    #     return test_dataloaders, val_dataloaders
    # elif test_dataloaders:
    #     return test_dataloaders
    # elif train_dataloaders:
    #     return train_dataloaders
    #Initialise data
    train_data = SPP_Dataset(train_path, target_col)
    train_dataloaders = DataLoader(train_data, 
                                    batch_size=batch_size, 
                                    shuffle=False)
    
    test_data = SPP_Dataset(test_path, target_col)
    test_dataloaders = DataLoader(test_data,
                                    batch_size=batch_size,
                                    shuffle=False)
    return train_dataloaders, test_dataloaders
    
def extractzip():
    """Set up folder and data for training and testing"""
    # Set up folder
    folder = os.path.join("..", "Data")
    train_folder = os.path.join(folder, "train")
    val_folder = os.path.join(folder, "validation")
    test_folder = os.path.join(folder, "test")

    # Check Data folder
    if os.path.exists(folder) and os.path.isdir(folder):
        print(f"[INFO] {folder} folder exists")
    else:
        print(f"[INFO] Did not find {folder}, creating one...")
        os.mkdir(folder)

    # Check train folder 
    if os.path.exists(train_folder) and os.path.isdir(train_folder):
        print(f"[INFO] {train_folder} folder exists")
    else:
        print(f"[INFO] Did not find {train_folder}, creating one...")
        os.mkdir(train_folder)

    # Check validation folder
    if os.path.exists(val_folder) and os.path.isdir(val_folder):
        print(f"[INFO] {val_folder} folder exists")
    else:
        print(f"[INFO] Did not find {val_folder}, creating one...")
        os.mkdir(val_folder)

    # Check test folder
    if os.path.exists(test_folder) and os.path.isdir(test_folder):
        print(f"[INFO] {test_folder} folder exists")
    else:
        print(f"[INFO] Did not find {test_folder}, creating one...")
        os.mkdir(test_folder)

    # Extract data file for training
    with zipfile.ZipFile(folder + "/data.zip", "r") as zip_train:
        print("[INFO] Unzipping training data")
        zip_train.extractall(train_folder)

    # Remove __MACOSX folder if it exists
    macosx_folder = os.path.join(train_folder, "__MACOSX")
    if os.path.exists(macosx_folder):
        print("[INFO] Removing __MACOSX folder")
        shutil.rmtree(macosx_folder)
 
    # Move contents from "STOCK PRICE DATA" to train folder
    stock_data_folder = os.path.join(train_folder, "STOCK PRICE DATA")
    if os.path.exists(stock_data_folder):
        print("[INFO] Moving files from 'STOCK PRICE DATA' to train folder")
        # Move all files from stock_data_folder to train_folder
        for file in os.listdir(stock_data_folder):
            src = os.path.join(stock_data_folder, file)
            dst = os.path.join(train_folder, file)
            shutil.move(src, dst)
        
        # Remove the now empty "STOCK PRICE DATA" folder
        print("[INFO] Removing empty 'STOCK PRICE DATA' folder")
        shutil.rmtree(stock_data_folder)
    
    #  # Extract zip file for validation
    # with zipfile.ZipFile(folder + "/val.zip", "r") as zip_val:
    #     print("[INFO] Unzipping validation data")
    #     zip_val.extractall(val_folder)

    # # Remove __MACOSX folder if it exists
    # macosx_folder = os.path.join(val_folder, "__MACOSX")
    # if os.path.exists(macosx_folder):
    #     print("[INFO] Removing __MACOSX folder")
    #     shutil.rmtree(macosx_folder)
 
    # # Move contents from "STOCK PRICE DATA" to val folder
    # stock_data_folder = os.path.join(val_folder, "STOCK PRICE DATA")
    # if os.path.exists(stock_data_folder):
    #     print("[INFO] Moving files from 'STOCK PRICE DATA' to val folder")
    #     # Move all files from stock_data_folder to val_folder
    #     for file in os.listdir(stock_data_folder):
    #         src = os.path.join(stock_data_folder, file)
    #         dst = os.path.join(val_folder, file)
    #         shutil.move(src, dst)
        
    #     # Remove the now empty "STOCK PRICE DATA" folder
    #     print("[INFO] Removing empty 'STOCK PRICE DATA' folder")
    #     shutil.rmtree(stock_data_folder)

    # # Extract zip file for testing
    # with zipfile.ZipFile(folder + "/test.zip", "r") as zip_test:
    #     print("[INFO] Unzipping training data")
    #     zip_test.extractall(test_folder)

    # # Remove __MACOSX folder if it exists
    # macosx_folder = os.path.join(test_folder, "__MACOSX")
    # if os.path.exists(macosx_folder):
    #     print("[INFO] Removing __MACOSX folder")
    #     shutil.rmtree(macosx_folder)

    # # Move contents from "Data Test" to train folder
    # stock_data_folder = os.path.join(test_folder, "Data Test")
    # if os.path.exists(stock_data_folder):
    #     print("[INFO] Moving files from 'Data Test' to test folder")
    #     # Move all files from stock_data_folder to test_folder
    #     for file in os.listdir(stock_data_folder):
    #         src = os.path.join(stock_data_folder, file)
    #         dst = os.path.join(test_folder, file)
    #         shutil.move(src, dst)
        
    #     # Remove the now empty "Data Test" folder
    #     print("[INFO] Removing empty 'Data Test' folder")
    #     shutil.rmtree(stock_data_folder)

def concat_files(
        folder_name: str) -> None:
    """Preprocesses and concatenates all files in a folder
    
    Args:
        folder_name: the name of folder whose files to be preprocessed and concatenated, "train" or "test" or "validation".
    """
    # Set up folder path
    folder = os.path.join("..","Data", folder_name)
    list_data = os.listdir(folder)
    preprocess_data = []
    # Loop through each dataset and preprocess
    for file in list_data:
        if file == '.DS_Store':
            os.remove(os.path.join(folder, file))
        else: 
            data = pd.read_csv(os.path.join(folder, file))
            data = preprocess(data)
            preprocess_data.append(data)
    # Concatenate into one dataset
    concat_data = pd.concat(preprocess_data, ignore_index=True)
    concat_data  = pd.get_dummies(concat_data, columns=['code'], dtype=float)
    concat_data = concat_data.fillna(0.0)

    # Remove file 
    shutil.rmtree(folder)
    os.mkdir(folder)
    print("[INFO] Removed all the files and recreated train folder")

    # Save file
    concat_data.to_csv(os.path.join(folder, "StockData_" + folder_name + ".csv"))
    print(f"[INFO] Saved file to {folder_name} folder")

    
