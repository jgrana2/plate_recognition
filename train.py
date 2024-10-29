from ultralytics import YOLO

# Load the model
backbone = YOLO('yolo11n.pt')

# Specify the dataset path correctly
dataset = 'datasets/data.yaml'

# Train the model
results = backbone.train(data=dataset, epochs=20)