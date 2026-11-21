import torch
import torch.nn as nn
from torchvision import models


class SpatialFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = "vgg16",
        num_classes: int = 8,
        feature_dim: int = 512,
        freeze_early_layers: bool = True
    ):
        super().__init__()
        if backbone.lower() == "vgg16":
            base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            if freeze_early_layers:
                # Gèle les 4 premiers blocs convolutifs (garde le bloc 5 entraînable)
                for param in list(base.features.parameters())[:-8]:
                    param.requires_grad = False
            self.features = base.features
            self.avgpool = base.avgpool
            in_features = 512 * 7 * 7
        else:
            raise ValueError(f"Backbone non supporté : {backbone}")

        self.projection = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projection(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)