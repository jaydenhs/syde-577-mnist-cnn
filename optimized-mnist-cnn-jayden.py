import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import sys
import datetime
import os
import matplotlib.pyplot as plt
import random
import numpy as np

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Redirect print output to a file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# make output directory
output_dir = f"mnist-cnn-jayden-outputs/{timestamp}"
try:
    os.makedirs(output_dir, exist_ok=True)
except OSError as e:
    print(f"Error creating directory {output_dir}: {e}")
    sys.exit(1)

output_file = f"{output_dir}/output.txt"
try:
    file = open(output_file, "w")
    sys.stdout = file
    file.write("")
except IOError as e:
    print(f"Error opening file {output_file}: {e}")
    sys.exit(1)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Best hyperparameters
best_params = {
    "lr": 0.0011981467283886237,
    "batch_size": 128,
    "dropout_rate": 0.36876369928643515,
    "l2_reg": 0.0007511356340641555,
    "num_conv_layers": 2,
    "num_filters": 32,
    "kernel_size": 3,
    "num_fc_layers": 2,
    "fc_units": 512,
    "rotation_degree": 10.701880875635522,
    "translate_range": 0.0043394023922881135,
    "scale_min": 0.8001053039300298,
    "scale_max": 1.1977000639362887,
}

# Data augmentation and normalization for training
transform = transforms.Compose(
    [
        transforms.RandomRotation(best_params["rotation_degree"]),
        transforms.RandomAffine(0, translate=(best_params["translate_range"], best_params["translate_range"])),
        transforms.RandomResizedCrop(28, scale=(best_params["scale_min"], best_params["scale_max"])),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Define a CNN architecture that has worked well for MNIST
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        layers = []
        in_channels = 1
        for _ in range(best_params["num_conv_layers"]):
            layers.append(
                nn.Conv2d(in_channels, best_params["num_filters"], kernel_size=best_params["kernel_size"], stride=1, padding=1)
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            in_channels = best_params["num_filters"]
        self.conv_layers = nn.Sequential(*layers)

        self.fc_layers = nn.ModuleList()
        conv_output_size = 28
        for _ in range(best_params["num_conv_layers"]):
            conv_output_size = (conv_output_size - best_params["kernel_size"] + 2 * 1) // 1 + 1  # Conv layer
            conv_output_size = (conv_output_size - 2) // 2 + 1  # MaxPool layer
        in_features = best_params["num_filters"] * conv_output_size * conv_output_size
        for _ in range(best_params["num_fc_layers"] - 1):
            self.fc_layers.append(nn.Linear(in_features, best_params["fc_units"]))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(best_params["dropout_rate"]))
            in_features = best_params["fc_units"]
        self.fc_layers.append(nn.Linear(in_features, 10))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x


model = CustomCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["l2_reg"])


def train(model, train_loader, criterion, optimizer, test_loader, epochs, patience):
    model.train()
    best_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(train_loader))
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Evaluate the model
        accuracy = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1}, Test Accuracy: {accuracy}")

        # Check for early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


def evaluate(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# Train and evaluate the model
train(model, train_loader, criterion, optimizer, test_loader, epochs=50, patience=10)
accuracy = evaluate(model, test_loader, criterion)
print(f"FINAL TEST ACCURACY: {accuracy}")

# Close the file
sys.stdout.close()
