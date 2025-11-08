import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import LinearClassifierModel

EPOCHS = 300
LEARNING_RATE = 0.1
BATCH_SIZE = 32
INPUT_FEATURES = 28*28
OUTPUT_FEATURES = 10
MODEL_SAVE_PATH = "model300e.pth"

device = "cpu"
print("Selected device:", device)

train_data = datasets.MNIST(
    root="data",
    train=True,    
    download=True,
    transform=ToTensor()
)


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


model = LinearClassifierModel(INPUT_FEATURES, OUTPUT_FEATURES)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

def train_model(model, epochs, optimizer, loss_fn, train_loader, test_loader):
    print(f"Initializing training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Training loss: {train_loss/len(train_loader):.4f}")
        
        train_loss /= len(train_loader)

    print("Training complete.")

    model.eval()
    test_loss = 0.0
    with torch.inference_mode():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            test_pred_logits = model(X_test_batch)
            loss = loss_fn(test_pred_logits, y_test_batch)
            test_loss += loss.item()
        test_loss /= len(test_loader)
    print(f"Test loss after training: {test_loss:.4f}")
    print("Training complete.")
    return model

if __name__ == "__main__":

    trained_model = train_model(model=model, 
                                epochs=EPOCHS, 
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                train_loader=train_loader,
                                test_loader=test_loader)


    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Modelo treinado foi salvo em: {MODEL_SAVE_PATH}")