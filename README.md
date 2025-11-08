# MNIST Handwritten Digit Recognition

This repository contains the source code for a handwritten digit recognition system using a linear classifier implemented in PyTorch, trained on the MNIST dataset. The project provides tools for training, batch prediction, and an interactive web-based interface for real-time digit recognition.

## 1. Introduction

The MNIST dataset is a cornerstone in the field of machine learning, comprising a large collection of handwritten digits used for training and evaluating image processing systems. This project implements a foundational linear classifier to recognize these digits, offering a clear and functional example of a complete machine learning workflow from training to interactive prediction.

## 2. System Components

The project is structured into several key components:

-   **`model.py`**: Defines the neural network architecture, a simple `LinearClassifierModel` that takes flattened 28x28 pixel images as input and outputs scores for each of the 10 digits (0-9).
-   **`train.py`**: A script for training the linear model. It utilizes the MNIST training dataset, Cross-Entropy Loss, and the SGD optimizer. Upon completion, it saves the trained model's state dictionary to a `.pth` file.
-   **`predict.py`**: A command-line script that loads a pre-trained model to perform predictions on a randomly selected grid of 9 images from the MNIST test set. The results, comparing predictions to true labels, are saved as a PNG image.
-   **`server.py`**: A Flask-based web server that exposes endpoints to list available models and to perform predictions on image data sent from a client.
-   **`index.html`**: A self-contained web application that provides a canvas for users to draw digits. It captures the drawing, sends it to the Flask server, and displays the model's prediction and activation flow in real-time.

## 3. How to Use

### 3.1. Prerequisites

-   Python 3.x
-   pip

### 3.2. Installation

1.  Clone the repository to your local machine.
2.  Install the required Python packages using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### 3.3. Training a New Model

1.  Run the training script. The script will automatically download the MNIST dataset to a `data/` directory.
    ```bash
    python train.py
    ```
2.  The trained model (e.g., `model300e.pth`) will be saved in the project's root directory. For the web interface to find it, create a `models/` directory and move the `.pth` file into it.

### 3.4. Running Predictions

There are two ways to perform predictions: via the command line for batch processing or through the interactive web interface.

#### Option 1: Command-Line Batch Prediction

This method is useful for quickly evaluating a model on a random set of test images. It does not require running the web server.

1.  Make sure you have a trained model file (e.g., `model80e.pth`) inside the `models/` directory. You may need to update the `MODEL_SAVE_PATH` variable in `predict.py` to point to your desired model.
2.  Execute the prediction script:
    ```bash
    python predict.py
    ```
3.  The output is an image file named `mnist_predictions.png` (by default) that shows a grid of digits with their predicted and true labels. The script randomly selects different digits from the test set each time it is run.

#### Option 2: Interactive Web Interface

This method provides a dynamic, real-time prediction experience by allowing you to draw digits and see the model's response instantly.

1.  Start the Flask web server:
    ```bash
    python server.py
    ```
2.  Once the server is running, open the `index.html` file in your web browser.
3.  Select an available model from the dropdown menu, draw a digit on the canvas, and the model's prediction will appear alongside a visualization of the neuron activations.

## 5. Results

After training and running the `predict.py` script, a visualization of the model's predictions on a random sample of 9 images from the test set is generated. The title of each subplot indicates the predicted label and the true label, with green for correct predictions and red for incorrect ones.

![MNIST Predictions](./mnist_predictions.png)

## 6. Project File Structure

```
/
├─── data/              # MNIST dataset (auto-downloaded)
├─── models/            # Directory for saved .pth model files
├─── .gitignore         # Git ignore file
├─── index.html         # Frontend for interactive predictions
├─── model.py           # Defines the PyTorch model architecture
├─── predict.py         # Script for batch predictions on test data
├─── README.md          # This file
├─── requirements.txt   # Python dependencies
├─── server.py          # Flask backend for serving predictions
└─── train.py           # Script for training the model
```