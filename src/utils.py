import torch

def get_device():
    """Returns GPU device if available; otherwise, CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, path):
    """Save model weights to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Load model weights from disk."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def accuracy(preds, labels):
    """Calculate the accuracy for a batch."""
    _, predicted = torch.max(preds, 1)
    return (predicted == labels).float().mean().item()
