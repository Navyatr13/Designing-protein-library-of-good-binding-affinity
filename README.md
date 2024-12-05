# PyTorch Sequence Classification

This project implements a modular and scalable sequence classification pipeline using PyTorch. It includes support for feedforward neural networks (FFNNs) and convolutional neural networks (CNNs). The script is highly modular, making it easy to extend for additional models and datasets.

---

## Features

- **Data Preprocessing**:
  - One-hot encoding of amino acid sequences.
  - Custom PyTorch Dataset for seamless data loading.
- **Modeling**:
  - Feedforward Neural Network (FFNN).
  - Convolutional Neural Network (CNN).
- **Training and Evaluation**:
  - Modular training and evaluation pipelines.
  - Binary classification metrics: accuracy, confusion matrix, and classification report.
- **Visualization**:
  - t-SNE visualization of cluster distributions.

---

## Prerequisites

### Libraries
- `torch`: For deep learning operations.
- `numpy`: For numerical computations.
- `pandas`: For data manipulation.
- `matplotlib`: For data visualization.
- `scikit-learn`: For splitting data and calculating metrics.

### Installation
Install required packages using pip:
``` pip install torch numpy pandas matplotlib scikit-learn ```

### File Structure
main.py: Main script containing the modularized PyTorch implementation.
Seq_Fitness_example.csv: Example dataset file (you need to provide this file).

### Dataset
The dataset should be a CSV file containing:

A Sequence column: The amino acid sequences.
A Fitness column: Target values for binary classification.

### Usage
#### Step 1: Data Preparation
Prepare your dataset (Seq_Fitness_example.csv) with sequences and fitness values.

#### Step 2: Run the Script
Run the main script:
``` python main.py ```
#### Step 3: Output
The script will:
Train the Feedforward Neural Network.
Train the Convolutional Neural Network.
Display training metrics and evaluation results (accuracy, confusion matrix, and classification report).
