import os
import torch

def print_config(config):
    for key, value in config.items():
        print(f"{key}: {value}")

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
