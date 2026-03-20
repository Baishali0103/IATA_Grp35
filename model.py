import torch
import torch.nn as nn
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,stride=2),
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2),
            nn.ReLU()
        )

        self._to_linear = None
        self._get_conv_output()

        self.fc = nn.Linear(self._to_linear, 9 * 3)

    def _get_conv_output(self):
        x = torch.randn(1, 3, 128, 128)
        x = self.cnn(x)
        self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, 9, 3)