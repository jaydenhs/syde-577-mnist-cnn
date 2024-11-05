import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna
from tqdm import tqdm
import sys
import datetime
import os
import matplotlib.pyplot as plt

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


# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.4)
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)

    # Network architecture hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3)
    num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    num_fc_layers = trial.suggest_int("num_fc_layers", 1, 3)
    fc_units = trial.suggest_categorical("fc_units", [128, 256, 512])

    # Data augmentation hyperparameters
    rotation_degree = trial.suggest_float("rotation_degree", 0, 20)
    translate_range = trial.suggest_float("translate_range", 0, 0.2)
    scale_min = trial.suggest_float("scale_min", 0.8, 1.0)
    scale_max = trial.suggest_float("scale_max", 1.0, 1.2)

    # Data augmentation and normalization for training
    transform = transforms.Compose(
        [
            transforms.RandomRotation(rotation_degree),
            transforms.RandomAffine(0, translate=(translate_range, translate_range)),
            transforms.RandomResizedCrop(28, scale=(scale_min, scale_max)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Define a CNN architecture that has worked well for MNIST
    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()

            # Convolutional layers
            layers = []
            in_channels = 1
            for _ in range(num_conv_layers):
                layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                # Update in_channels for the next iteration
                in_channels = num_filters
            self.conv_layers = nn.Sequential(*layers)

            # Calculate the number of features after the convolutional layers
            conv_output_size = 28
            for _ in range(num_conv_layers):
                conv_output_size = (conv_output_size - kernel_size + 2 * 1) // 1 + 1  # Conv layer
                conv_output_size = (conv_output_size - 2) // 2 + 1  # MaxPool layer
            in_features = num_filters * conv_output_size * conv_output_size

            # Fully connected layers
            self.fc_layers = nn.ModuleList()
            for _ in range(num_fc_layers - 1):
                self.fc_layers.append(nn.Linear(in_features, fc_units))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(dropout_rate))
                in_features = fc_units

            # Add the final output layer
            self.fc_layers.append(nn.Linear(in_features, 10))

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            for layer in self.fc_layers:
                x = layer(x)
            return x

    model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

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
    return accuracy


# Create a study and optimize the objective function with tqdm progress bar
study = optuna.create_study(direction="maximize")
for _ in tqdm(range(100), desc="Optuna Trials"):
    study.optimize(objective, n_trials=1)

# Print the best hyperparameters and best test accuracy
print("Best hyperparameters: ", study.best_params)
print("Best test accuracy: ", study.best_value)

# Plot and save the hyperparameter importances
optuna.visualization.matplotlib.plot_param_importances(study)
plt.savefig(f"{output_dir}/param_importances.png")

# Close the file
sys.stdout.close()
