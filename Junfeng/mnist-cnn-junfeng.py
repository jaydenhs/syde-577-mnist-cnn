import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna
from tqdm import tqdm
import datetime
import os
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, Any


# Setup logging
def setup_logging(output_dir: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(f"{output_dir}/training.log"), logging.StreamHandler()],
    )


# Configure output directory
def setup_output_dir() -> Tuple[str, str]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"mnist-cnn-junfeng-outputs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, f"{output_dir}/output.txt"


class CustomCNN(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.conv_layers = self._build_conv_layers(config)
        self.fc_layers = self._build_fc_layers(config)

    def _build_conv_layers(self, config: Dict[str, Any]) -> nn.Sequential:
        layers = []
        in_channels = 1
        conv_output_size = 28

        for _ in range(config["num_conv_layers"]):
            layers.extend(
                [
                    nn.Conv2d(in_channels, config["num_filters"], kernel_size=config["kernel_size"], padding="same"),
                    nn.BatchNorm2d(config["num_filters"]),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(config["dropout_rate"]),
                ]
            )
            in_channels = config["num_filters"]
            conv_output_size //= 2

        return nn.Sequential(*layers)

    def _build_fc_layers(self, config: Dict[str, Any]) -> nn.Sequential:
        layers = []
        in_features = config["num_filters"] * (28 // (2 ** config["num_conv_layers"])) ** 2

        for _ in range(config["num_fc_layers"] - 1):
            layers.extend(
                [
                    nn.Linear(in_features, config["fc_units"]),
                    nn.BatchNorm1d(config["fc_units"]),
                    nn.ReLU(),
                    nn.Dropout(config["dropout_rate"]),
                ]
            )
            in_features = config["fc_units"]

        layers.append(nn.Linear(in_features, 10))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        return self.fc_layers(x)


def objective(trial: optuna.Trial) -> float:
    config = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.5),
        "l2_reg": trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True),
        "num_conv_layers": trial.suggest_int("num_conv_layers", 2, 4),
        "num_filters": trial.suggest_categorical("num_filters", [32, 64, 128]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        "num_fc_layers": trial.suggest_int("num_fc_layers", 2, 4),
        "fc_units": trial.suggest_categorical("fc_units", [128, 256, 512]),
    }

    # Data transformations with better augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Dataset and loaders
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform_train)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2_reg"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    # Train the model
    best_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(50):  # Max epochs
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        accuracy = evaluate(model, test_loader, device)
        scheduler.step(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:  # Early stopping
                break

    return best_accuracy


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def main():
    output_dir, output_file = setup_output_dir()
    setup_logging(output_dir)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    logging.info(f"Best parameters: {study.best_params}")
    logging.info(f"Best accuracy: {study.best_value:.2f}%")

    # Save results
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(f"{output_dir}/param_importances.png")


if __name__ == "__main__":
    main()
