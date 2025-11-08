import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import LinearClassifierModel

INPUT_FEATURES = 28*28
OUTPUT_FEATURES = 10
MODEL_SAVE_PATH = "models/model80e.pth"

device = "cpu"

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=False, 
    transform=ToTensor()
)

model_to_load = LinearClassifierModel(INPUT_FEATURES, OUTPUT_FEATURES)
model_to_load.load_state_dict(torch.load(MODEL_SAVE_PATH))
model_to_load.to(device)

print("Model loaded. Starting predictions...")

def plot_predictions(model, data, save_path="test.png"):
    # here I'll plot 9 random predictions from the test set
    plt.figure(figsize=(10,10))
    plt.suptitle("Model Prediction vs. Ground Truth", fontsize=16, y=1.02)

    rng = np.random.default_rng()
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

    plot_predictions(model_to_load, test_data)