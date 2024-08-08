import torch
from ultralytics import YOLO
from torchvision import transforms

class YOLOv8sPose:
    def __init__(self, modelconfig_path, model_weights='yolov8s-pose.pt'):
        self.model = YOLO(modelconfig_path, model_weights=model_weights)

    def train_model(self, dataconfig_path, num_epochs=10, learning_rate=0.001):
        self.model.train(data=dataconfig_path, epochs=num_epochs, lr0=learning_rate)

    def predict(self, image_path, best_model_path):
        self.model = YOLO(best_model_path)  # Load the best model weights
        results = self.model.predict(source=image_path)
        return results