import torch
import torch.nn as nn
from TestFunction import test_cnn_model
from ENV import MiniGridCNNEnv

from train import ActorCriticCNNPolicy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = "DoorKey-16x16_policy.pth"
    policy = ActorCriticCNNPolicy().to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    print(f"Model loaded from {model_path}")

    
    test_cnn_model(
        model=policy,
        env_class=MiniGridCNNEnv,
        env_name="MiniGrid-DoorKey-16x16-v0",
        action_type="discrete",
        test_N=10,
        save=True,
        modelname=ActorCriticCNNPolicy.__name__
    )