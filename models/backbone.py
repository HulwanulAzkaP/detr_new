import torch
import torch.nn as nn
import torchvision.models as models
from config.config import CONFIG

class DETRBackbone(nn.Module):
    def __init__(self, pretrained=CONFIG["model"]["pretrained"]):
        super(DETRBackbone, self).__init__()
        num_classes = CONFIG["dataset"]["num_classes"]

        # Load ResNet as a backbone
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the fully connected (FC) layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC and avgpool layers

        # Positional Encoding for DETR compatibility
        self.position_embedding = nn.Conv2d(2048, 256, kernel_size=1)

        # Transformer (encoder-decoder)
        self.transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

        # Class and bounding box prediction heads
        self.class_head = nn.Linear(256, num_classes)
        self.bbox_head = nn.Linear(256, 4)

    def forward(self, x):
        # Backbone (ResNet features)
        features = self.backbone(x)  # [batch_size, 2048, H/32, W/32]

        # Flatten spatial dimensions and apply positional encoding
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, -1).permute(2, 0, 1)  # [H*W, batch_size, channels]
        pos_encoded_features = self.position_embedding(features.view(batch_size, channels, height, width))

        # Transformer
        transformed = self.transformer(pos_encoded_features.view(height * width, batch_size, 256),
                                        pos_encoded_features.view(height * width, batch_size, 256))

        # Predict classes and bounding boxes
        classes = self.class_head(transformed.mean(dim=0))  # Average over spatial dimensions
        bboxes = self.bbox_head(transformed.mean(dim=0))    # Bounding box prediction

        return classes, bboxes
