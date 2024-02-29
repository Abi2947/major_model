import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from urllib import urlopen

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset path
data_dir = "/home/avinash/AG/VSC/hand_gesture/split_data"
train_dir ="/home/avinash/AG/VSC/hand_gesture/split_data/train"
test_dir ="/home/avinash/AG/VSC/hand_gesture/split_data/test"

# Define  the transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])

#Load the original dataset
original_dataset = ImageFolder(root = data_dir, transform = transforms)

#Print the class name
class_names = original_dataset.classes
print("Class Name:", class_names)

# Define the Bidirectional LSTM model
class BiLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate parameters
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate parameters
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        # Cell parameters
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate parameters
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        bs, _ = x.size()

        h_t, c_t = (torch.zeros(self.hidden_size).to(x.device),
                    torch.zeros(self.hidden_size).to(x.device)) if init_states is None else init_states

        # Input gate
        i_t = torch.sigmoid(x @ self.W_ii.t() + self.b_ii + h_t @ self.W_hi.t() + self.b_hi)
        # Forget gate
        f_t = torch.sigmoid(x @ self.W_if.t() + self.b_if + h_t @ self.W_hf.t() + self.b_hf)
        # Cell gate
        g_t = torch.tanh(x @ self.W_ig.t() + self.b_ig + h_t @ self.W_hg.t() + self.b_hg)
        # Output gate
        o_t = torch.sigmoid(x @ self.W_io.t() + self.b_io + h_t @ self.W_ho.t() + self.b_ho)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(BiLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm_forward = nn.ModuleList([BiLSTMCell(input_size, hidden_size) for _ in range(num_layers)])
        self.lstm_backward = nn.ModuleList([BiLSTMCell(input_size, hidden_size) for _ in range(num_layers)])

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        bs, seq_len, _ = x.size()

        h_t_forward, c_t_forward = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        h_t_backward, c_t_backward = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)

        for t in range(seq_len):
            # Forward pass
            for layer in range(self.num_layers):
                h_t_forward, c_t_forward = self.lstm_forward[layer](x[:, t, :], (h_t_forward, c_t_forward))

            # Backward pass
            for layer in range(self.num_layers):
                h_t_backward, c_t_backward = self.lstm_backward[layer](x[:, seq_len - t - 1, :], (h_t_backward, c_t_backward))

        h_t = torch.cat([h_t_forward, h_t_backward], dim=1)
        out = self.fc(h_t)

        return out

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
tloss = []
tacc = []

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = outputs.max(1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    epochs_list.append(epoch + 1)
    tloss.append(loss.item())
    tacc.append(accuracy)


# Evaluation on the test set
model.eval()
correct = 0
total = 0
vloss = 0.0
vacc = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Compute accuracy
        _, val_preds = torch.max(outputs, 1)
        correct += (val_preds == labels).sum().item()
        total += labels.size(0)

        vloss += loss.item()

    # Calculate validation accuracy and loss
    vacc = correct / total
    vloss /= len(test_loader)

    print(f'Validation Loss: {vloss:.4f}, Accuracy: {100 * vacc:.2f}%')


# Plotting Tranning and validation looss curve over epochs
plt.plot(tloss,'-o')
plt.plot(vloss,'-o')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Loss')
plt.grid(True)

plt.show()

# Plotting Tranning and validation accuracies curve over epochs
plt.plot(tacc,'-o')
plt.plot(vacc,'-o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Accuracy')
plt.grid(True)

plt.show()