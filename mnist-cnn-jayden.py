import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna
from tqdm import tqdm

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    l2_reg = trial.suggest_loguniform('l2_reg', 1e-5, 1e-2)

    # Data augmentation and normalization for training
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(128 * 3 * 3, 256)
            self.fc2 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(dropout_rate)
            self.batch_norm1 = nn.BatchNorm2d(32)
            self.batch_norm2 = nn.BatchNorm2d(64)
            self.batch_norm3 = nn.BatchNorm2d(128)
            self.batch_norm_fc = nn.BatchNorm1d(256)
            self.l2_reg = l2_reg

        def forward(self, x):
            x = self.pool(self.batch_norm1(torch.relu(self.conv1(x))))
            x = self.pool(self.batch_norm2(torch.relu(self.conv2(x))))
            x = self.pool(self.batch_norm3(torch.relu(self.conv3(x))))
            x = x.view(-1, 128 * 3 * 3)
            x = self.dropout(self.batch_norm_fc(torch.relu(self.fc1(x))))
            x = self.fc2(x)
            return x

    model = CustomCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=model.l2_reg)

    def train(model, train_loader, criterion, optimizer, epochs=10):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    def evaluate(model, test_loader, criterion):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    # Train and evaluate the model
    train(model, train_loader, criterion, optimizer, epochs=10)
    accuracy = evaluate(model, test_loader, criterion)
    return accuracy

# Create a study and optimize the objective function with tqdm progress bar
study = optuna.create_study(direction='maximize')
for _ in tqdm(range(5), desc='Optuna Trials'):
    study.optimize(objective, n_trials=1)

# Print the best hyperparameters and best test accuracy
print('Best hyperparameters: ', study.best_params)
print('Best test accuracy: ', study.best_value)