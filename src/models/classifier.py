"""
ConvNeXt-Tiny classifier via timm with custom head.
"""

import torch
import torch.nn as nn
import timm
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str = "convnext_tiny"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.1


class WatermarkClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = timm.create_model(
            config.name, pretrained=config.pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(feat_dim, config.num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

    @classmethod
    def from_config(cls, config_dict: dict):
        return cls(ModelConfig(**config_dict))