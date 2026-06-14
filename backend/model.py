import torch.nn as nn
import torchvision.models as models

class GCPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        features = self.backbone(x)

        coords = self.regression_head(features)
        shape = self.classification_head(features)

        return coords, shape