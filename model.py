import torch
import torch.nn as nn
from torchvision import models


class MultiHeadOutput(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.obj1_size_head = nn.Linear(feature_dim, 2)
        self.obj1_color_head = nn.Linear(feature_dim, 4)
        self.obj1_shape_head = nn.Linear(feature_dim, 3)
        self.relation_head = nn.Linear(feature_dim, 4)
        self.obj2_size_head = nn.Linear(feature_dim, 2)
        self.obj2_color_head = nn.Linear(feature_dim, 4)
        self.obj2_shape_head = nn.Linear(feature_dim, 3)

    def forward(self, x):
        return {
            "obj1_size": self.obj1_size_head(x),
            "obj1_color": self.obj1_color_head(x),
            "obj1_shape": self.obj1_shape_head(x),
            "relation": self.relation_head(x),
            "obj2_size": self.obj2_size_head(x),
            "obj2_color": self.obj2_color_head(x),
            "obj2_shape": self.obj2_shape_head(x),
        }


class CNNMultiHeadSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.heads = MultiHeadOutput(256)

    def forward(self, x):
        x = self.features(x)
        x = self.shared_fc(x)
        return self.heads(x)


class CNNMultiHeadDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.heads = MultiHeadOutput(256)

    def forward(self, x):
        x = self.features(x)
        x = self.shared_fc(x)
        return self.heads(x)


class ResNet18MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.heads = MultiHeadOutput(256)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared_fc(x)
        return self.heads(x)