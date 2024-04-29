import os
import yaml
import torch
from pathlib import Path
from yolov5.train import train
from yolov5.val import val
from yolov5.detect import run

def prepare_dataset(data_dir):
    # Create dataset.yaml
    data_yaml = {
        "train": os.path.join(data_dir, "train"),
        "val": os.path.join(data_dir, "val"),
        "nc": 1,
        "names": ["path"]
    }
    with open("dataset.yaml", "w") as f:
        yaml.dump(data_yaml, f)

def train_model(weights, data, img_size, batch_size, epochs):
    # Train the model
    train(data=data, imgsz=img_size, weights=weights, batch_size=batch_size, epochs=epochs)

def evaluate_model(weights, data, img_size, conf_thres):
    # Evaluate the model on the test set
    val(data=data, weights=weights, imgsz=img_size, conf_thres=conf_thres)

def run_inference(weights, source, img_size, conf_thres):
    # Run inference on the specified source (image, video, or webcam)
    run(weights=weights, source=source, imgsz=img_size, conf_thres=conf_thres)

def main():
    # Set the paths and parameters
    data_dir = "path/to/dataset"
    weights = "yolov5s.pt"
    img_size = 640
    batch_size = 16
    epochs = 100
    conf_thres = 0.4

    # Prepare the dataset
    prepare_dataset(data_dir)

    # Train the model
    train_model(weights, "dataset.yaml", img_size, batch_size, epochs)

    # Evaluate the model on the test set
    best_weights = "runs/train/exp/weights/best.pt"
    evaluate_model(best_weights, "dataset.yaml", img_size, conf_thres)

    # Run inference on a test image
    test_image = "path/to/test/image.jpg"
    run_inference(best_weights, test_image, img_size, conf_thres)

    # Run inference on the webcam
    run_inference(best_weights, "0", img_size, conf_thres)

if __name__ == "__main__":
    main()