import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class BackBone(nn.Module):
    def __init__(self, backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=0):
        super(BackBone, self).__init__()
        self.backbone = resnet_fpn_backbone(backbone_name=backbone_name, weights=weights, trainable_layers=trainable_layers)
    
    def forward(self, x):
        output = self.backbone(x)

        return output["1"],output["2"],output["3"]

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    input_data = torch.randn(1, 3, 640, 640)
    
    # Initialize the BackBone model
    backbone_model = BackBone()
    
    # Forward pass
    output = backbone_model(input_data)
    
    # Print the shapes of the selected outputs
    for v in output:
        print(f"{v.shape}")
