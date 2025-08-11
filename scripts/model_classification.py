import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out) * x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Applies channel attention then spatial attention.
    """
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DualEfficientNetClassifier(nn.Module):
    """
    Dual-input classifier using two EfficientNet-B0 backbones (clinical + Wood's lamp).
    Optionally applies CBAM attention after feature fusion.
    Outputs a single logit for progressive vs stable classification.
    """
    def __init__(self, use_cbam=False, pretrained=True):
        super(DualEfficientNetClassifier, self).__init__()
        # Load pretrained EfficientNet-B0 models
        backbone1 = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone2 = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove original classifier heads
        self.encoder1 = backbone1.features
        self.encoder2 = backbone2.features
        # Global pooling to get 1280-D feature vectors
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Attention
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(channels=1280 * 2, ratio=16)
        # Classification head: input dim = 1280*2, hidden=512, output=1 logit
        self.classifier = nn.Sequential(
            nn.Linear(1280 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

    def forward(self, img_clinical, img_wood):
        # Extract feature maps
        f1 = self.encoder1(img_clinical)  # [B,1280,H,W]
        f2 = self.encoder2(img_wood)      # [B,1280,H,W]
        # Global average pool
        v1 = self.global_pool(f1).view(f1.size(0), -1)  # [B,1280]
        v2 = self.global_pool(f2).view(f2.size(0), -1)  # [B,1280]
        # Concatenate feature vectors
        fused = torch.cat([v1, v2], dim=1)  # [B,2560]
        # Optional CBAM on fused feature (reshape to  [B,2560,1,1])
        if self.use_cbam:
            fused_feat_map = fused.view(fused.size(0), -1, 1, 1)
            fused_feat_map = self.cbam(fused_feat_map)
            fused = fused_feat_map.view(fused.size(0), -1)
        # Classification logit
        logit = self.classifier(fused)  # [B,1]
        return logit.squeeze(1)  # return [B]
