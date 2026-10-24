import pandas as pd
import sklearn
import torch
import streamlit as st
df=pd.read_csv("breast_cancer.csv")
# print(df.head())
# print(df.info())
# print("Dataset shape:", df.shape)

# print("\nMissing values:")
# print(df.isnull().sum())

# print("\nDuplicate rows:", df.duplicated().sum())

# print("\nStatistical summary:")
# print(df.describe())

# graph 
import matplotlib.pyplot as plt

df["diagnosis"].value_counts().plot(
    kind="bar",
    title="Distribution of Diagnosis"
)



plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.tight_layout()
# plt.show()
# plt.savefig("graph.png")
# print("Graph saved successfully!")
# Remove completely empty columns
df = df.dropna(axis=1, how="all")

# Separate features and target
X = df.drop(columns=["diagnosis", "id"], errors="ignore")

# Convert diagnosis into numeric labels
y = df["diagnosis"].map({"B": 0, "M": 1})


# splitting data for training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# print("Training samples:", len(X_train))
# print("Testing samples:", len(X_test))
from sklearn.impute import SimpleImputer
# Handle missing values
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# scaling data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# applying models
# linear regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)

print(
    "Accuracy:",
    accuracy_score(y_test, lr_predictions)
)
# random forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, rf_predictions)
)

# neural network
# coverting data into pytorch
import torch
from torch.utils.data import TensorDataset, DataLoader

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

import numpy as np
import torch

y_train_tensor = torch.tensor(
    y_train.to_numpy().astype(np.float32).reshape(-1, 1),
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.to_numpy().astype(np.float32).reshape(-1, 1),
    dtype=torch.float32
)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
# defining neural
import torch.nn as nn

class BreastCancerNN(nn.Module):

    def __init__(self, input_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# Create the model OUTSIDE the class
model = BreastCancerNN(
    input_features=X_train_tensor.shape[1]
)

print(model)

print(model)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
epochs = 100

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for batch_X, batch_y in train_loader:

        # Forward propagation
        logits = model(batch_X)

        # Calculate loss
        loss = criterion(logits, batch_y)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item() * len(batch_X)

    average_loss = total_loss / len(train_dataset)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.4f}"
        )


    # Save model and preprocessing files
    import os
import joblib
import torch
os.makedirs("model", exist_ok=True)

torch.save(model.state_dict(), "model/cancer_model.pth")

joblib.dump(imputer, "model/imputer.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(X.columns.tolist(), "model/features.pkl")

print("All model files saved successfully!")