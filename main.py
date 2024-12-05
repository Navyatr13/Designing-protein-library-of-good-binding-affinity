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

from utils import *

def main():
    # Load and preprocess data
    file_path = "Seq_Fitness_example.csv"
    X, y = load_and_preprocess_data(file_path)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create PyTorch datasets and loaders
    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train and evaluate Feedforward Neural Network
    input_dim = X_train.shape[1]
    ffnn = FeedforwardNN(input_dim)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(ffnn.parameters(), lr=0.001)

    print("Training Feedforward Neural Network...")
    train_model(ffnn, criterion, optimizer, train_loader, num_epochs=20)

    print("Evaluating Feedforward Neural Network...")
    evaluate_model(ffnn, test_loader)

    # Train and evaluate CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], 1, -1)
    X_test_cnn = X_test.reshape(X_test.shape[0], 1, -1)

    train_dataset_cnn = SequenceDataset(X_train_cnn, y_train)
    test_dataset_cnn = SequenceDataset(X_test_cnn, y_test)

    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=32, shuffle=False)

    cnn = ConvNet(input_channels=1)
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)

    print("Training Convolutional Neural Network...")
    train_model(cnn, criterion, optimizer_cnn, train_loader_cnn, num_epochs=20)

    print("Evaluating Convolutional Neural Network...")
    evaluate_model(cnn, test_loader_cnn)


if __name__ == "__main__":
    main()
