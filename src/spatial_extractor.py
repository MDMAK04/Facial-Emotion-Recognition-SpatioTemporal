import torch
import torch.nn as nn
from torchvision import models

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "vgg16", num_classes: int = 8, feature_dim: int = 512, freeze_early_layers: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == "vgg16":
            base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            if freeze_early_layers:
                for param in list(base.features.parameters())[:17]:
                    param.requires_grad = False
            self.features = base.features
            self.avgpool = base.avgpool
            in_features = 512 * 7 * 7

        elif self.backbone_name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            if freeze_early_layers:
                for name, param in base.named_parameters():
                    if "layer4" not in name and "fc" not in name:
                        param.requires_grad = False
            self.features = nn.Sequential(*list(base.children())[:-1])
            self.avgpool = nn.Identity()
            in_features = 2048

        elif self.backbone_name == "vit":
            base = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            if freeze_early_layers:
                for name, param in base.named_parameters():
                    if "encoder.layers.encoder_layer_11" not in name and "heads" not in name:
                        param.requires_grad = False
            base.heads = nn.Identity()
            self.features = base
            self.avgpool = nn.Identity()
            in_features = 768
        else:
            raise ValueError("Backbone non supporté : 'vgg16', 'resnet50', 'vit'")

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.projection(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))