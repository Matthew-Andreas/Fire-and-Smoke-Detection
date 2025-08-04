from ultralytics import YOLO

# Load trained model from Phase 1
model = YOLO("runs/detect/train39/weights/last.pt")

# Unfreeze all layers
for p in model.model.parameters():
    p.requires_grad = True

# Train full model
model.train(
    data="fire_data.yaml",
    epochs=290,
    imgsz=640,
    batch=128,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=0,
    warmup_bias_lr=0.01,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    flipud=0.0, fliplr=0.0,
    scale=0.0, degrees=0.0, translate=0.0, shear=0.0,
    workers=32,
    device='0,1,2,3',
)

# Validate the model
model.val(data="fire_data.yaml", split='test')  # Make sure it's validating on the correct dataset split

# Print the path to the latest trained model weights
latest_run = list(Path("runs/detect/").glob("train*"))[-1]
print(latest_run)