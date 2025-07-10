from ultralytics import YOLO
from pathlib import Path

# Define YAML configuration for dataset
data_yaml = """
train: /home/andre150/CS478Project/fireDataset/train
val: /home/andre150/CS478Project/fireDataset/valid
test: /home/andre150/CS478Project/fireDataset/test

nc: 2
names: ['fire', 'smoke']
"""

# Save the YAML configuration
with open("fire_data.yaml", "w") as f:
    f.write(data_yaml)

# Load YOLO model(8 or 12) 
model = YOLO("yolov12s.pt")# Add whichever model you are using here 

# Set the device to use multiple GPUs (e.g., GPU 0 and 1)
device = '0,1,2,3'  # List of GPU IDs, or 'cuda' to use all available GPUs

# Train the model with custom parameters
model.train(
    data="fire_data.yaml",         # Dataset YAML
    epochs=300,                     # Total epochs (as per the paper)
    imgsz=640,                     # Image size (640x640)
    batch=128,                      # Reduced batch size for memory efficiency
    optimizer="SGD",               # Use SGD optimizer
    lr0=0.01,                      # Initial learning rate
    momentum=0.937,                # Momentum parameter
    weight_decay=0.0005,           # Weight decay
    warmup_epochs=3,               # Warm-up for the first 3 epochs
    warmup_bias_lr=0.01,           # Final warm-up learning rate (0.01)
    
    # Disable augmentations (per the paper’s specification)
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,   # Disable color augmentations
    flipud=0.0, fliplr=0.0,             # Disable flipping augmentations
    scale=0.0, degrees=0.0, translate=0.0, shear=0.0,  # Disable geometric & affine augmentations
    
    workers=16,                    # Adjust the number of workers
    device=device                  # Specify the devices (GPUs) to use
)

# Validate the model
model.val(data="fire_data.yaml", split='test')  # Make sure it's validating on the correct dataset split

# Print the path to the latest trained model weights
latest_run = list(Path("runs/detect/").glob("train*"))[-1]
print(latest_run)
