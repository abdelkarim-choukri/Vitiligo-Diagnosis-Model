import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DualInputSegmentationUNet(nn.Module):
    """
    Dual-input UNet segmentation using EfficientNet-B0 encoder.
    Concatenates clinical and Wood's lamp images as 6-channel input.
    Outputs a single-channel mask (logits) for vitiligo lesion segmentation.

    Requires `pip install segmentation-models-pytorch`.
    """
    def __init__(self, encoder_name='efficientnet-b0', encoder_weights='imagenet'):
        super(DualInputSegmentationUNet, self).__init__()
        # Create a UNet with EfficientNet-B0 backbone, expecting 6-channel input
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=6,
            classes=1,
            activation=None  # output raw logits
        )

    def forward(self, img_clinical, img_wood):
        # img_clinical, img_wood shape: [B,3,H,W]
        # Concatenate along channel dimension to get [B,6,H,W]
        x = torch.cat([img_clinical, img_wood], dim=1)
        mask_logits = self.unet(x)
        return mask_logits.squeeze(1)  # return [B,H,W]
