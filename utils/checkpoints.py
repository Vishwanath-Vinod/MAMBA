from pathlib import Path
import torch
from pathlib import Path

def save_df(df,path,):
    """
    Creates parent directories automatically and creates csv files.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_checkpoint(model,optimizer,epoch,best_val_loss,path,):
    """
    Save a training checkpoint containing the model state, optimizer state, current epoch, and best validation loss.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"epoch": epoch,"best_val_loss": best_val_loss,"model_state_dict": model.state_dict(),"optimizer_state_dict": optimizer.state_dict(),}
    torch.save(checkpoint, path)


def load_checkpoint(model,optimizer,path,device=None,):
    """
    Load a training checkpoint and restore the model and optimizer states.
    start_epoch: epoch to resume from
    best_val_loss: best validation loss seen so far
    """

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    return start_epoch, best_val_loss


def load_model_weights(model, path,device=None,):
    """
    Load only model weights.Useful for evaluation or inference.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model