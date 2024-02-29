# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# class_labels = ['Aaja', 'Dhanyabadh', 'Ghar', 'Hami', 'Janxu', 'Ma', 'Namaskar']  # Define your hand sign labels

# # Define your model architecture if necessary
# class ViTModel(nn.Module):
#     def __init__(self, num_classes):
#         super(ViTModel, self).__init__()
#         self.vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)

#         # Remove the original fully connected head
#         self.vit_model.head = nn.Identity()

#         # Adapt the final fully connected layer to match the number of classes
#         self.fc = nn.Linear(self.vit_model.embed_dim, num_classes)

#     def forward(self, x):
#         x = self.vit_model(x)
#         x = self.fc(x)
#         return x

# # Function to preprocess the image
# def preprocess_image(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
#     image = cv2.resize(image, (224, 224))  # Resize the image to match the input size of the model
#     image = np.array(image) / 255.0  # Normalize pixel values
#     return image

# # Load the trained model
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = ViTModel(num_classes=7) # Initialize your model
#     model.load_state_dict(torch.load('../hand_gesture/test.pth', map_location=torch.device('cpu')))  # Load your trained model here
#     model.eval()  # Set the model to evaluation mode
#     return model

# # Main function to run the app
# def main():
#     st.title("Hand Sign Recognition")

#     start_camera = st.checkbox('Start Camera')

#     if start_camera:
#         cap = cv2.VideoCapture(0)  # Open default camera (0)

#         model = load_model()

#         while cap.isOpened():
#             ret, frame = cap.read()  # Read frame from camera
#             if not ret:
#                 st.write('Error: Cannot read frame')
#                 break

#             frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect

#             # Display the frame
#             stframe = st.empty()
#             stframe.image(frame, channels="RGB")

#             # Preprocess the frame and make predictions
#             preprocessed_frame = preprocess_image(frame,class_labels)
#             preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
#             preprocessed_frame = torch.tensor(preprocessed_frame.transpose(0, 3, 1, 2)).float()  # Transpose and convert to tensor
#             prediction = model(preprocessed_frame)
#             # Apply softmax if necessary
#             prediction = F.softmax(prediction, dim=1)
#             predicted_class = torch.argmax(prediction, dim=1).item()

#             # Show the prediction
#             st.write(f"Predicted Class: {class_labels[predicted_class]}")

#             if not start_camera:
#                 break

#         cap.release()

# if __name__ == '__main__':
#     main()

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import threading

class_labels = ['Aaja', 'Dhanyabadh', 'Ghar', 'Hami', 'Janxu', 'Ma', 'Namaskar']

class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        self.vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit_model.head = nn.Identity()
        self.fc = nn.Linear(self.vit_model.embed_dim, num_classes)

    def forward(self, x):
        x = self.vit_model(x)
        x = self.fc(x)
        return x

def preprocess_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return frame

def camera_capture(cap, model):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Error: Cannot read frame')
            break

        frame = cv2.flip(frame, 1)

        preprocessed_frame = preprocess_image(frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        preprocessed_frame = torch.tensor(preprocessed_frame).float()

        prediction = model(preprocessed_frame)
        predicted_class = torch.argmax(prediction, dim=1).item()

        cv2.putText(frame, f"Predicted Class: {class_labels[predicted_class]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)
    model = ViTModel(num_classes=7)
    model.load_state_dict(torch.load('../hand_gesture/test.pth', map_location=torch.device('cpu')))
    model.eval()

    thread = threading.Thread(target=camera_capture, args=(cap, model))
    thread.start()

if __name__ == '__main__':
    main()
