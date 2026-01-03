import torch

def pick_device() -> str:
    # Apple Silicon uses MPS when available
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
