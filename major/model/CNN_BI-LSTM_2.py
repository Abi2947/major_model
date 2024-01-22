# !pip install torch
# !pip install torchtext==0.6.0
# !pip install torchvision
# !pip install spacy

import spacy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torchtext.data import Field, BucketIterator, Example
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from spacy.cli.download import download
download(model="en_core_web_sm")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset path
data_dir = "/home/abinash/AG/VSC_projects/major/Dataset"

# Define Bi-LSTM model for sentence generation
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Define CNN model for gesture recognition
class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 16 * 16, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.fc(x)
        return x

# Define dataset class for hand gestures with 7 classes
class GestureDataset(Dataset):
    def __init__(self, data_folder, transform=None):
            self.dataset = ImageFolder(root=data_folder, transform=transform)

    def __getitem__(self, index):
            return self.dataset[index]

    def __len__(self):
            return len(self.dataset)

# Define NLP processing using torchtext
spacy.load("en_core_web_sm")
TEXT = Field(tokenize='spacy', lower=True, include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False)

# Hyperparameters
input_size = 64  # Adjustments based on our input data size
hidden_size = 128
num_classes = 7  # Assuming 7 classes
num_layers = 2
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Data preprocessing and loading
train_dataset = ImageFolder(root = data_dir, transform = transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = ImageFolder(root = data_dir, transform = transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# Initialize the model, loss function, and optimizer
model = BiLSTM(input_size, hidden_size, num_classes, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
model = model.to(device)
epochs_list = []
loss_list = []
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    epochs_list.append(epoch + 1)
    loss_list.append(loss.item())

# Plotting the loss over epochs
plt.plot(epochs_list, loss_list, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Validation
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')