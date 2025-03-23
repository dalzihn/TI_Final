import torch
import torch.nn
import torch.utils.data 
import os
from datetime import datetime
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

#Reference: 
# 1. https://www.learnpytorch.io/05_pytorch_going_modular/
# 2. https://www.learnpytorch.io/07_pytorch_experiment_tracking/
def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> tuple[float, float]:
    """Performs training step of a model on a single epoch
    
    Args:
        model: a Pytorch model
        dataloader: a Pytorch DataLoader that will be used for training
        loss_func: loss function of training step
        optimizer: optimization technique of the training step
    Returns:
        A tuple of training loss and evaluation score
        In the form (training_loss, training_evalscore)."""
    # Put model in train mode
    model.to(device)
    model.train()

    # Initialise training loss and evaluation score
    training_loss = 0
    eval_score = 0 #Currently, it is RMSE

    # Loop through data to train the model
    for batch, (X, y) in enumerate(dataloader):
        # Step 1: forward pass
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        y_pred = y_pred.squeeze(dim=1)

        # Step 2: calculate loss
        loss = loss_func(y_pred, y)
        training_loss += loss
        eval_score += loss

        # Step 3: optimizer zero_grad step
        optimizer.zero_grad()

        # Step 4: backpropagation
        loss.backward()

        # Step 5: optimizer step
        optimizer.step()
    # Normalize training loss and evaluation score
    training_loss /= (len(dataloader)*dataloader.batch_size)
    eval_score = (eval_score / (len(dataloader)*dataloader.batch_size))**(1/2)
    
    return training_loss, eval_score

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn,
        device: torch.device
) -> tuple[float, float]:
    """Performs testing step on a single epoch
    Args:
        model: a Pytorch model
        dataloader: a Pytorch Daloader that will be used for testing
        loss_func: loss function of testing step
    
    Returns:
        A tuple that shows testing loss and evaluation score
        In the form (testing_loss, eval_score)"""
    
    # Put model in evaluation mode
    model.to(device)
    model.eval()

    # Initialise testing loss and evaluation score
    testing_loss = 0
    eval_score = 0
    prediction = []
    #Turn on context manager
    with torch.inference_mode():
        # Loop through data to test model
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # Step 1: forward pass
            y_pred = model(X)
            y_pred = y_pred.squeeze(dim=1)
            prediction.append(y_pred)
            # Step 2: calculate loss
            loss = loss_func(y_pred, y)
            testing_loss += loss

            # Step 3: calculate evaluation
            eval_score += loss
        testing_loss = testing_loss.item()
        eval_score = eval_score.item()

    #Normalize metrics  
    testing_loss /= (len(dataloader)*dataloader.batch_size)
    eval_score = (eval_score / (len(dataloader)*dataloader.batch_size))**(1/2)

    return prediction, testing_loss, eval_score

# def validate_step(
#         model: torch.nn.Module,
#         dataloader: torch.utils.data.DataLoader,
#         loss_func: torch.nn,
#         device: torch.device
# ) -> tuple[float, float]:
#     """Performs testing step on a single epoch
#     Args:
#         model: a Pytorch model
#         dataloader: a Pytorch Daloader that will be used for validation
#         loss_func: loss function of validating step
    
#     Returns:
#         A tuple that shows validating loss and evaluation score
#         In the form (testing_loss, eval_score)"""
    
#     # Put model in evaluation mode
#     model.to(device)
#     model.eval()

#     # Initialise testing loss and evaluation score
#     val_loss = 0
#     eval_score = 0
#     #Turn on context manager
#     with torch.inference_mode():
#         # Loop through data to test model
#         for batch, (X, y) in enumerate(dataloader):
#             X, y = X.to(device), y.to(device)
#             # Step 1: forward pass
#             y_pred = model(X)
#             y_pred = y_pred.squeeze(dim=1)
#             # Step 2: calculate loss
#             loss = loss_func(y_pred, y)
#             val_loss += loss

#             # Step 3: calculate evaluation
#             eval_score += loss
#         val_loss = val_loss.item()
#         eval_score = eval_score.item()

#     #Normalize metrics  
#     val_loss /= (len(dataloader)*dataloader.batch_size)
#     eval_score = (eval_score / (len(dataloader)*dataloader.batch_size))**(1/2)
    
#     return val_loss, eval_score

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        writer: torch.utils.tensorboard.writer.SummaryWriter = None,
) -> torch.nn.Module:
    """Trains a deep learning model via many epochs
    
    Args:
        model: a Pytorch model
        train_daloader: a Pytorch Dataloader that will be used for training
        test_dataloader: a Pytorch Dataloader that will be used for tesdting
        loss_func: loss function for training
        optimizer: optimization technique for training
        
    Returns:
        A dictionary.
        In the form {"train_loss": [],
                     "train_score": [],
                     "test_loss": [],
                     "test_score": []}."""
    #Loop through data to train
    results = {}
    train_loss = []
    train_score = []
    test_loss = []
    test_score = []
    for epoch in tqdm(range(epochs)):
        train_loss_scalar, train_score_scalar = train_step(model=model, 
                                                           dataloader=train_dataloader, 
                                                           loss_func=loss_func, 
                                                           optimizer=optimizer,
                                                           device=device)
        _, test_loss_scalar, test_score_scalar = test_step(model=model,
                                                           dataloader=test_dataloader,
                                                           loss_func=loss_func,
                                                           device=device)
        train_loss.append(train_loss_scalar.item())
        train_score.append(train_score_scalar.item())
        test_loss.append(test_loss_scalar)
        test_score.append(test_score_scalar)
    
        print(
            f"Epoch: {epoch + 1} | "
            f"Train RMSE: {train_score_scalar:.4f} | "
            f" Test RMSE: {test_score_scalar:.4f}")
        
        # NOTE: Experiment tracking
        if writer is not None:
            writer.add_scalar(tag="Loss_train",
                            scalar_value=train_loss,
                            global_step=epoch)
            
            writer.add_scalar(tag="Loss_test",
                            scalar_value=test_loss,
                            global_step=epoch)
            
            
            writer.close()
    results['train_loss'] = train_loss
    results['train_score'] = train_score
    results['test_loss'] = test_loss
    results['test_score'] = test_score
    return results

def create_writer(
        experiment_name: str,
        model_name: str,
        misc: str=None
)  -> torch.utils.tensorboard.writer.SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter for saving results of training/testing model

    The format will be: log/YYYY-MM-DD/experiment_name/model_name/misc
    Args:
        experiment_name: name of the experiment
        model_name: name of the model
        misc: extra things to be added
        
    Returns:
        An instance of torch.utils.tensorboard.writer.SummaryWriter"""
    
    time = datetime.now().strftime("%Y-%m-%d")

    if misc:
        log_dir = os.path.join("..", "log", time, experiment_name, model_name, misc)
    else:
        log_dir = os.path.join("..", "log", time, experiment_name, model_name)

    print(f"[INFO] An instane of SummaryWriter is created, saving to: {log_dir}")

    return SummaryWriter(log_dir=log_dir)
    
def save_model(
        model: torch.nn.Module,
        model_name: str,
        save_folder: os.path,
) -> os.path:
    """Saves a Pytorch model
    
    Args:
        model: the model to be saved
        model_name: name of the model
        save_folder: the folder used to store model

    Returns:
        A os.path which is the path to the model"""
    path = os.path.join(save_folder, model_name)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model is saved successfully to {path}")
    return path