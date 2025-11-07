# I don't have a GPU, so I'll always use CPU. **It's great to notice that pip install -r requirements.txt will download the whole PyTorch package including CUDA support even if you don't have a GPU.**

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor 
from torch.utils.data import DataLoader

device = "cpu"
print("Selected device:", device)

train_data = datasets.MNIST(
    root="data",
    train=True,    
    download=True,
    transform=ToTensor() 
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class LinearClassifierModel(nn.Module):
    def __init__(self, input_feat, output_feat):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(input_feat, output_feat)

    def forward(self, x):
        x_flattened = self.flatten(x)
        return self.linear_layer(x_flattened)

loss_fn = nn.CrossEntropyLoss()

def train_model(model, epochs, optimizer):
    print(f'Starting training for {epochs} epochs...')
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)

            loss = loss_fn(y_pred_logits, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)

        model.eval()
        test_loss = 0.0
        with torch.inference_mode():
            for X_test_batch, y_test_batch in test_dataloader:
                X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)

                y_test_logits = model(X_test_batch)

                loss += loss_fn(y_test_logits, y_test_batch).item()

            test_loss /= len(test_dataloader)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")

    print("Training complete.")
    return model

def plot_predictions(model, data, save_path="mnist_predictions.png"):
    # here I'll plot 9 random predictions from the test set
    plt.figure(figsize=(10,10))
    plt.suptitle("Model Prediction vs. Ground Truth", fontsize=16, y=1.02)

    rng = np.random.default_rng(42)
    indices =rng.choice(len(data), size=9, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i+1)
        img, label = data[idx]

        model.eval()
        with torch.inference_mode():
            pred_logits = model(img[None, ...].to(device))
        
        pred_label = torch.argmax(pred_logits, dim=1).item()
        color = "g" if pred_label == label else "r"

        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Pred: {pred_label}, True: {label}", color=color)
        plt.axis(False)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved prediction plot to {save_path}")

if __name__ == "__main__":

    torch.manual_seed(42)


    EPOCHS = 20
    LEARNING_RATE = 0.1
    INPUT_FEATURES = 28*28
    OUTPUT_FEATURES = 10   

    model = LinearClassifierModel(input_feat=INPUT_FEATURES, 
                                  output_feat=OUTPUT_FEATURES)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    trained_model = train_model(model=model, 
                                epochs=EPOCHS, 
                                optimizer=optimizer)

    plot_predictions(model=trained_model, 
                           data=test_data)