import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=10000, hidden_dim=512, num_classes=90):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)