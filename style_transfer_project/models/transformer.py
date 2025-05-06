import torch.nn as nn

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
