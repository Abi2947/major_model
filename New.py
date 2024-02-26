from flask import Flask, render_template, Response
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

app = Flask(__name__)

# Define your model architecture if necessary
class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        self.vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)

        # Remove the original fully connected head
        self.vit_model.head = nn.Identity()

        # Adapt the final fully connected layer to match the number of classes
        self.fc = nn.Linear(self.vit_model.embed_dim, num_classes)

    def forward(self, x):
        x = self.vit_model(x)
        x = self.fc(x)
        return x

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    image = cv2.resize(image, (224, 224))  # Resize the image to match the input size of the model
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

# Load the trained model
def load_model():
    model = ViTModel(num_classes=7) # Initialize your model with 7 output classes
    model.load_state_dict(torch.load('test.pth', map_location=torch.device('cpu')))  # Load your trained model here
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define class labels
class_labels = ['Aaja', 'Dhanyabadh', 'Ghar', 'Hami', 'Janxu', 'Ma', 'Namaskar']

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the camera feed
def gen_frames():
    cap = cv2.VideoCapture(0)  # Open default camera (0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()  # Read frame from camera
        if not ret:
            print("Error: Unable to capture frame.")
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect

        # Preprocess the frame and make predictions
        preprocessed_frame = preprocess_image(frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
        preprocessed_frame = torch.tensor(preprocessed_frame.transpose(0, 3, 1, 2)).float()  # Transpose and convert to tensor
        prediction = model(preprocessed_frame)
        prediction = F.softmax(prediction, dim=1)
        predicted_class = torch.argmax(prediction, dim=1).item()

        # Draw prediction text on frame
        prediction_text = class_labels[predicted_class]
        cv2.putText(frame, prediction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Route for the camera feed page
@app.route('/camera_feed')
def camera_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
