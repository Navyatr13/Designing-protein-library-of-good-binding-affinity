import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
class SequenceDataset(Dataset):
    """
    Custom Dataset for loading sequence data.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(file_path, sequence_col="Sequence", target_col="Fitness"):
    """
    Load data from CSV and preprocess for PyTorch.
    """
    df = pd.read_csv(file_path)

    # One-hot encode sequences
    AAlist = np.array(list("ACDEFGHIKLMNPQRSTVWXYZ_-"))
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(AAlist.reshape(-1, 1))

    def encode_sequence(sequence):
        return encoder.transform(np.array(list(sequence)).reshape(-1, 1)).flatten()

    df["One_Hot"] = df[sequence_col].apply(encode_sequence)
    X = np.array(df["One_Hot"].tolist())
    y = np.array(df[target_col])
    return X, y


def split_data(X, y, test_size=0.3):
    """
    Split data into training and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


class FeedforwardNN(nn.Module):
    """
    Feedforward Neural Network.
    """
    def __init__(self, input_dim, hidden_dim=90):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ConvNet(nn.Module):
    """
    Convolutional Neural Network.
    """
    def __init__(self, input_channels=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, activation='relu')
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, activation='relu')
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def train_model(model, criterion, optimizer, train_loader, num_epochs=20):
    """
    Train a PyTorch model.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")


def evaluate_model(model, test_loader):
    """
    Evaluate a PyTorch model.
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.numpy())
            all_targets.extend(y_batch.numpy())

    print("Accuracy:", accuracy_score(all_targets, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))
    print("Classification Report:\n", classification_report(all_targets, all_preds))
