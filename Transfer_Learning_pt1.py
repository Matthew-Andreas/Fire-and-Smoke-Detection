from ultralytics import YOLO
import torch.nn as nn
from ultralytics.nn.modules.head import Detect

# Load base model
model = YOLO("yolov8s.pt")# Add whichever model you are using here 
base_model = model.model

class CNNEnhancedHead(nn.Module):
    def __init__(self, nc, ch):
        super().__init__()
        # Example CNN block for the last feature map
        self.conv_block = nn.Sequential(
            nn.Conv2d(ch[2], 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ch[2], kernel_size=1)  # project back to original channels
        )
        self.detect = Detect(nc=nc, ch=ch)

    def forward(self, x):
        x[2] = self.conv_block(x[2])  # Modify only the last input to Detect
        return self.detect(x)

# Replace the Detect head with the custom one
new_head = CNNEnhancedHead(nc=2, ch=[128, 256, 512])
for i, m in enumerate(base_model.model):
    if isinstance(m, Detect):
        base_model.model[i] = new_head
        break

# Freeze backbone
for name, param in base_model.named_parameters():
    param.requires_grad = False
for name, param in new_head.named_parameters():
    param.requires_grad = True

# Assign model and train head only
model.model = base_model
model.train(
    data="fire_data.yaml",
    epochs=10,
    imgsz=640,
    batch=128,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_bias_lr=0.01,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    flipud=0.0, fliplr=0.0,
    scale=0.0, degrees=0.0, translate=0.0, shear=0.0,
    workers=16,
    device='0,1,2,3'
)