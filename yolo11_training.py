from ultralytics import YOLO
import torch

def main():
 
    # loading a model to utilize for the training, trying a pretrained small model to see how it affects training
    model = YOLO("yolo11s.yaml")

    # from this we can train the model on our data now utilizing model.train()
    # hsv_v value of either being brighter 60% or darker 60%
    # translate will translate image in a random direction by 0.2
    # scale scales the image within a range of 0.5 to help with further away detection
    # shear changes the persepctive of the image to help with objects that might have different perspectives
    # mosaic create a collage of 4 images stitched together to help with dense object detection scenarios
    # mosaic close stops the possibility of mosaic from happening to let the model train on raw data towards the end of the epochs
    results = model.train(data = "data.yaml", epochs = 100, imgsz = 960, hsv_v = 0.6, translate = 0.2, scale = 0.6, shear = 0.1, mosaic = 0.5, close_mosaic = 20) 

if __name__ == "__main__":
    
    main()