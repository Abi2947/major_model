import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset path
dataset_path = "/content/drive/MyDrive/Dataset"

#Load the original dataset
original_dataset = ImageFolder(root = dataset_path, transform = transforms)

#Print the class name
class_names = original_dataset.classes
print("Class Name:", class_names)

# Define the CNN + BiLSTM model
class BILSTM(nn.Module):
    def __init__(self, num_classes):
        super(BILSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Custom dataset class (assuming you have organized your data in folders)
class Data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir,transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# Hyperparameters
num_classes = 7
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset
train_dataset = Data(root_dir=dataset_path, transform=transform)
test_dataset = Data(root_dir=dataset_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = BILSTM(num_classes)
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