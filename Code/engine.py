import torch
import torch.nn
import torch.utils.data 
from tqdm.auto import tqdm

#Reference: https://www.learnpytorch.io/05_pytorch_going_modular/
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

            # Step 2: calculate loss
            loss = loss_func(y_pred, y)
            testing_loss += loss

            # Step 3: calculate evaluation
            eval_score += loss

    #Normalize metrics  
    testing_loss /= len(dataloader)
    eval_score = torch.sqrt(eval_score) / len(dataloader)
    
    return testing_loss, eval_score

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn,
        epochs: int,
        optimizer: torch.optim.Optimizer
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
        tracking[str(epoch)] = [training_loss, eval_score]
    
        print(
            f"Epoch: {epoch + 1} | "
            f"Train loss: {training_loss:.4f} | "
            f"Train evaluation score: {eval_score:.4f}")
    return tracking

def test(
        model: torch.nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        loss_func: torch.nn,
        epochs: int,
        optimizer: torch.optim.Optimizer
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
    for epoch in tqdm(epochs):
        testing_loss, eval_score = test_step(model=model, 
                                             dataloader=test_dataloader, 
                                             loss_func=loss_func)
        tracking[str(epoch)] = [testing_loss, eval_score]
    
        print(
            f"Epoch: {epoch + 1} | "
            f"Train loss: {testing_loss:.4f}"
            f"Train evaluation score: {eval_score:.4f}")
    return tracking

        


    
