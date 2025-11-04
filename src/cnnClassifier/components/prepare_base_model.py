import os
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        # Load VGG16 pretrained model
        try:
            # Try new API (torchvision >= 0.13)
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except (TypeError, AttributeError):
            # Fall back to old API (torchvision < 0.13)
            self.model = models.vgg16(pretrained=True)
        
        # Remove the classifier (last few layers)
        self.model = nn.Sequential(*list(self.model.features.children()))
        
        self.save_model(path=self.config.base_model_path, model=self.model)

    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            # Freeze layers up to freeze_till
            for i, param in enumerate(model.parameters()):
                if i < freeze_till:
                    param.requires_grad = False

        # Add classifier head
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, classes)
        )

        full_model = nn.Sequential(model, classifier)
        
        # Print model summary
        print(full_model)
        return full_model
    
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
        
    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)

