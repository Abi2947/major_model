import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Define class labels
class_labels = ['Aaja', 'Dhanyabadh', 'Ghar', 'Hami', 'Janxu', 'Ma', 'Namaskar']

# Define the architecture of the model
class HandGestureModel(nn.Module):
    def __init__(self, num_classes):
        super(HandGestureModel, self).__init__()
        # Define your model architecture here
        # Example:
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*56*56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Define the forward pass of your model
        # Example:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*56*56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        pass

# Load the pre-trained model
def load_model(model_path):
    model = HandGestureModel(num_classes=len(class_labels))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Capture frames from the camera and perform prediction
def predict_gesture(model, camera_id=0):
    cap = cv2.VideoCapture(camera_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Cannot read frame')
            break

        # Preprocess the frame
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = preprocess_image(image)

        # Perform prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            prediction = class_labels[predicted.item()]

        # Display the prediction on the frame
        cv2.putText(frame, f"Prediction: {prediction}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the pre-trained model
    model_path = '/home/avinash/AG/VSC/hand_gesture/test.pth'
    model = load_model(model_path)

    # Predict hand gestures in real-time
    predict_gesture(model)
