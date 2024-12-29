from torch import optim
from torch import nn
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from model import GPAPredictor


def load_data(file_path):
    gpa = ["GPA1", "GPA2"]
    df = pd.read_csv(file_path)
    X = df[gpa].values
    y = (
        (df["GPA2"] + np.random.normal(0, 0.1, df.shape[0]))
        .clip(0, 10)
        .values.reshape(-1, 1)
    )
    names = df["name"].values
    return X, y, names


def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=123
    )
    scaler_X, scaler_y = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(
        feature_range=(0, 1)
    )

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.fit_transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.fit_transform(y_val)

    X_train = torch.FloatTensor(X_train_scaled)
    X_val = torch.FloatTensor(X_val_scaled)
    y_train = torch.FloatTensor(y_train_scaled)
    y_val = torch.FloatTensor(y_val_scaled)

    model = GPAPredictor()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    epochs = 5000
    best_val_loss = float("inf")
    patience = 200
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_func(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = loss_func(val_outputs, y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "models/gpa_model.pth")
        else:
            counter += 1
        if counter > patience:
            print(f"Early stopping at each {epoch+1}")
            break
        if (epoch + 1) % patience == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
            )

        model.load_state_dict(torch.load("models/gpa_model.pth"))
        return model, scaler_X, scaler_y


def predict_gpa(model, scaler_X, scaler_y, name, X, names):
    if name in names:
        index = list(names).index(name)
        student_data = X[index]
        student_data_scaled = scaler_X.transform([student_data])
        input_tensor = torch.FloatTensor(student_data_scaled)

        model.eval()
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
        prediction = scaler_y.inverse_transform(prediction_scaled.numpy())
        return prediction[0][0]
    else:
        return None


def main():
    file_path = "data/data.csv"
    X, y, names = load_data(file_path)
    model, scaler_X, scaler_y = train_model(X, y)
    while True:
        name = input("Enter student name (or 'q' to quit):")
        if name.lower == "q":
            break
        predicted_gpa = predict_gpa(model, scaler_X, scaler_y, name, X, names)
        if predicted_gpa is not None:
            print(f"Predicted next GPA for {name}: {predicted_gpa:.2f}")
        else:
            print(f"Student {name} not found in the database")


if __name__ == "__main__":
    main()
