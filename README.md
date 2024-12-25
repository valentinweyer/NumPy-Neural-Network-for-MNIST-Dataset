# NumPy Neural Network for MNIST Dataset

A pure NumPy implementation of a deep neural network for classifying handwritten digits from the MNIST dataset. This project demonstrates how to build a neural network from scratch using only NumPy, implementing forward propagation, backpropagation, and various activation functions.

## Features

- Pure NumPy implementation of a deep neural network
- Three-layer architecture (784->128->64->10)
- ReLU and Softmax activation functions
- Cross-entropy loss function
- Mini-batch gradient descent
- Data normalization and preprocessing
- Training/validation split
- Accuracy evaluation

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
pillow
idx2numpy
torch  # Only used for input preprocessing
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Dataset

The MNIST dataset needs to be downloaded from [Kaggle's Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data). After downloading:

1. Extract the downloaded files
2. Place `train.csv` and `test.csv` in the `/data/raw` directory

## Project Structure

```
├── data
│   ├── external       # Data from third party sources
│   ├── interim        # Intermediate data
│   ├── processed      # Final, canonical data sets
│   └── raw           # Original, immutable data dump
├── models            # Trained models
├── notebooks         # Jupyter notebooks
├── references        # Data dictionaries, manuals, etc.
├── reports          
│   └── figures       # Generated graphics and figures
├── src
│   └── NN_Numpy.py   # Source code for the neural network
├── requirements.txt  # Project dependencies
└── README.md
```

## Implementation Details

The neural network implementation includes:

- **FCLayer**: Fully connected layer implementation with forward and backward passes
- **NeuralNetwork**: Main class that combines layers and implements:
  - Forward propagation
  - Backward propagation
  - Various activation functions (ReLU, Sigmoid, Softmax)
- **Utils**: Helper functions for data preprocessing and visualization

### Network Architecture

1. Input Layer: 784 neurons (28x28 pixel images)
2. First Hidden Layer: 128 neurons with ReLU activation
3. Second Hidden Layer: 64 neurons with ReLU activation
4. Output Layer: 10 neurons with Softmax activation (one for each digit)

## Usage

1. Ensure the MNIST dataset is placed in `/data/raw`
2. Run the neural network:

```bash
python src/NN_Numpy.py
```

The script will:
- Load and preprocess the MNIST data
- Train the neural network for 100 epochs
- Display training loss for each epoch
- Evaluate the model on validation data
- Show example predictions with visualizations

## Results

The model achieves reasonable accuracy on the validation set while being implemented purely in NumPy. Training progress is displayed during execution:

```
Epoch 1/100, Loss: X.XXXX
...
Epoch 100/100, Loss: X.XXXX
Validation Accuracy: XX.XX%
```

Example predictions are visualized using matplotlib, showing the original image and the model's prediction.

## License

This project is licensed under the terms of the LICENSE file included in the repository.