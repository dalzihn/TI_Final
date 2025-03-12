import torch
import torch.nn
import torch.utils.data 
import os
from datetime import datetime
import torch.utils.tensorboard
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

#Reference: 
# 1. https://www.learnpytorch.io/05_pytorch_going_modular/
# 2. https://www.learnpytorch.io/07_pytorch_experiment_tracking/
def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn,
        optimizer: torch.optim.Optimizer
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
    model.train()

    # Initialise training loss and evaluation score
    training_loss = 0
    eval_score = 0 #Currently, it is RMSE

    # Loop through data to train the model
    for batch, (X, y) in enumerate(dataloader):
        # Step 1: forward pass
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
    training_loss /= len(dataloader)
    eval_score = torch.sqrt(eval_score) / len(dataloader)
    
    return training_loss, eval_score

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn,
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
    model.eval()

    # Initialise testing loss and evaluation score
    testing_loss = 0
    eval_score = 0

    #Turn on context manager
    with torch.inference_mode():
        # Loop through data to test model
        for batch, (X, y) in enumerate(dataloader):
            # Step 1: forward pass
            y_pred = model(X)
            y_pred = y_pred.squeeze(dim=1)

            # Step 2: calculate loss
            loss = loss_func(y_pred, y)
            testing_loss += loss

            # Step 3: calculate evaluation
            eval_score += loss
        testing_loss = testing_loss.item()
        eval_score = eval_score.item()

    #Normalize metrics  
    testing_loss /= len(dataloader)
    eval_score = (eval_score**(1/2)) / len(dataloader)
    
    return testing_loss, eval_score

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        writer: torch.utils.tensorboard.writer.SummaryWriter
) -> torch.nn.Module:
    """Trains a deep learning model via many epochs
    
    Args:
        model: a Pytorch model
        train_daloader: a Pytorch Dataloader that will be used for training
        loss_func: loss function for training
        optimizer: optimization technique for training
        
    Returns:
        A dictionary with epoch as key and training loss, evaluation score as value 
        In the form {epoch: [training_loss, eval_score]}"""
    #Loop through data to train
    tracking = {}
    for epoch in tqdm(range(epochs)):
        training_loss, eval_score = train_step(model=model, 
                                               dataloader=train_dataloader, 
                                               loss_func=loss_func, 
                                               optimizer=optimizer)
        tracking[str(epoch)] = [training_loss.item(), eval_score.item()]
    
        print(
            f"Epoch: {epoch + 1} | "
            f"Train loss: {training_loss:.4f} | "
            f"Train evaluation score: {eval_score:.4f}")
        
        # NOTE: Experiment tracking
        writer.add_scalar(tag="Loss_train",
                          scalar_value=training_loss,
                          global_step=epoch)
        
        writer.add_scalar(tag="EvaluationScore_train",
                          scalar_value=eval_score,
                          global_step=epoch)
        
        
        writer.close()

    return tracking

def test(
        model: torch.nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn,
        epochs: int,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
) -> torch.nn.Module:
    """Tests a deep learning model via many epochs
    
    Args:
        model: a Pytorch model
        test_daloader: a Pytorch Dataloader that will be used for testing
        loss_func: loss function for testing
        optimizer: optimization technique for testing
        
    Returns:
        A dictionary with epoch as key and testing loss, evaluation score as value 
        In the form {epoch: [testing_loss, eval_score]}"""
    #Loop through data to train
    tracking = {}
    for epoch in tqdm(range(epochs)):
        testing_loss, eval_score = test_step(model=model, 
                                             dataloader=test_dataloader, 
                                             loss_func=loss_func)
        tracking[str(epoch)] = [testing_loss, eval_score]
    
        print(
            f"Epoch: {epoch + 1} | "
            f"Test loss: {testing_loss:.4f} | "
            f"Test evaluation score: {eval_score:.4f}")
        
        # NOTE: Experiment tracking
        writer.add_scalar(tag="Loss_test",
                          scalar_value=testing_loss,
                          global_step=epoch)
        
        writer.add_scalar(tag="EvaluationScore_test",
                          scalar_value=eval_score,
                          global_step=epoch)
        
        
        writer.close()
    return tracking

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
        log_dir = os.path.join("..", "log", experiment_name, model_name, misc)
    else:
        log_dir = os.path.join("..", "log", experiment_name, model_name)

    print(f"[INFO] An instane of SummaryWriter is created, saving to: {log_dir}")

    return SummaryWriter(log_dir=log_dir)
    
