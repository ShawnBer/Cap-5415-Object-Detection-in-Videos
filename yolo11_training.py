from ultralytics import YOLO
import torch

def main():
 
    # loading a model to utilize for the training
    model = YOLO("yolo11m.yaml")

    # from this we can train the model on our data now utilizing model.train()
    results = model.train(data = "data.yaml", epochs = 50, imgsz = 640)

if __name__ == "__main__":
    main()