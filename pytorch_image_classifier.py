import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

# Load the pre-trained ResNet50 model from ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# Specify the image path directly
img_path = 'download.jpg'  # Replace with your image's correct path

# Check if the image exists at the given path
if not os.path.exists(img_path):
    print("The image does not exist at the provided path. Please check the path.")
else:
    # Load the image
    img = Image.open(img_path)

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Classify the image
    with torch.no_grad():
        output = model(img_tensor)

    # Load the class labels from a JSON file (imagenet_classes.json)
    with open("imagenet_classes.json") as f:
        labels = json.load(f)

    # Get the predicted class index and label
    _, predicted_idx = torch.max(output, 1)  # Corrected here to get the index
    predicted_label = labels[str(predicted_idx.item())]

    # Print the result
    print(f"The image is classified as: {predicted_label}")
