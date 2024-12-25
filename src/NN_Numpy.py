from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os 


script_dir = os.path.dirname(os.path.abspath(__file__))

X_train = pd.read_csv(os.path.join(script_dir, "../data/raw/train.csv"))
X_train = X_train.drop(columns="label")
y_train = pd.read_csv(os.path.join(script_dir,"../data/raw/train.csv"), usecols=["label"])

# Split train.csv into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = X_train / 255.0  # Normalize pixel values to [0, 1]
X_val = X_val / 255.0      # Normalize pixel values to [0, 1]

class NeuralNetwork():
    def __init__(self):
        self.layer1 = FCLayer(784, 128, "relu")  # Match PyTorch input size
        self.layer2 = FCLayer(128, 64, "relu")
        self.layer3 = FCLayer(64, 10, "softmax")
        
    def forward(self, x):
        layer1_output = self.layer1.forward(x)
        layer2_output = self.layer2.forward(layer1_output)
        layer3_output = self.layer3.forward(layer2_output)
        
        return layer3_output
    
    def backward(self, dj_da, learning_rate):
        """
        Backward pass through all layers.
        dj_da: Gradient of the loss w.r.t. the network's output.
        learning_rate: Learning rate for gradient descent.
        """
        dj_da = self.layer3.backward(dj_da, learning_rate)
        dj_da = self.layer2.backward(dj_da, learning_rate)
        dj_da = self.layer1.backward(dj_da, learning_rate)
    
    @staticmethod
    def linear(z):
        return z
    
    
    @staticmethod
    def relu(z):
        return np.maximum(z, 0)
    
    @staticmethod
    def relu_back(z):
        if z < 0:
            o = 0
        elif z > 0:
            o = 1
        else:
            o = None
        
        return o

    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        # Subtract the maximum value from each row for numerical stability
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(z_stable)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_back(z):
        z_vector = z.reshape(z.shape[0],1)
        z_matrix = np.tile(z_vector, z.shape[0])    
        print(z_matrix, '\n')
        print(np.transpose(z_matrix))
        np.diag(z) - (z_matrix * np.transpose(z_matrix))
        

class FCLayer():
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size)
                
    def forward(self, x):
        self.x = x
        
        self.z = np.dot(self.x, self.weights) + self.bias
        
        if self.activation == "relu":
            self.output = NeuralNetwork.relu(self.z)
        
        elif self.activation == "softmax":
            self.output = NeuralNetwork.softmax(self.z)
            
        return self.output 
    
    def backward(self, dj_da, learning_rate):
        # Compute dj_dz based on the activation function
        if self.activation == "relu":
            dj_dz = dj_da * (self.z > 0)  # ReLU derivative
        elif self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-self.z))
            dj_dz = dj_da * sig * (1 - sig)
        elif self.activation == "softmax":
            dj_dz = dj_da  # Softmax gradient already computed

        # Compute gradients w.r.t weights, biases, and inputs
        batch_size = self.x.shape[0]
        dj_dw = np.dot(self.x.T, dj_dz) / batch_size
        dj_db = np.sum(dj_dz, axis=0, keepdims=True) / batch_size

        # Dynamically adjust shape of dj_db to match self.bias
        self.bias -= learning_rate * dj_db.flatten()  # Flatten dj_db to match shape of bias
        self.weights -= learning_rate * dj_dw        # Update weights

        # Return gradient to propagate backward to the previous layer
        dj_dx = np.dot(dj_dz, self.weights.T)
        return dj_dx

class utils():
    def preprocess_input(X_data, target_size=784):
        # Ensure input is a NumPy array
        if isinstance(X_data, pd.DataFrame):
            X_data = X_data.values
        elif isinstance(X_data, list):
            X_data = np.array(X_data, dtype=np.float32)
        
        # Ensure data is of type float32
        X_data = X_data.astype(np.float32)
        
        n_samples, original_features = X_data.shape
        if original_features < target_size:
            # Pad with zeros if input size is smaller than the target
            padding = target_size - original_features
            X_data = np.pad(X_data, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        elif original_features > target_size:
            # Truncate if input size is larger than the target
            X_data = X_data[:, :target_size]
        
        return X_data
    
    
    @staticmethod
    # Function to display an image
    def show_image(image, title=None, subtitle=None):
        plt.imshow(image, cmap="gray")
        
        if title:
            plt.title(title, fontsize=14, y=1.02)  # Main title above the image
        
        if subtitle:
            plt.text(
                0.5, -0.1, subtitle, 
                fontsize=12, color="black", 
                ha="center", va="center", 
                transform=plt.gca().transAxes  # Position relative to the axes
            )
        
        plt.axis("off")
        plt.show()

def one_hot_encode(y, num_classes):
    # Ensure y is a NumPy array
    y = y.to_numpy() if isinstance(y, pd.DataFrame) else y

    batch_size = y.shape[0]
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), y.flatten()] = 1
    return one_hot

model = NeuralNetwork()


epochs = 100
learning_rate = 0.1

for epoch in range(epochs):
    # Forward pass
    outputs = model.forward(X_train)

    # Convert y_train to one-hot encoding dynamically
    num_classes = outputs.shape[1]
    y_one_hot = one_hot_encode(y_train, num_classes)

    # Compute loss (Cross-Entropy Loss)
    batch_size = y_train.shape[0]
    loss = -np.sum(y_one_hot * np.log(outputs + 1e-9)) / batch_size

    # Compute initial gradient
    dj_da = outputs - y_one_hot

    # Backward pass
    model.backward(dj_da, learning_rate)

    # Print loss
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Preprocess the validation set
X_val_processed = utils.preprocess_input(X_val)

# Convert to NumPy for the NumPy model
X_val_processed_np = X_val_processed

# Forward pass on validation set
val_outputs = model.forward(X_val)

# Compute accuracy
val_preds = np.argmax(val_outputs, axis=1)  # Predicted class indices
val_labels = y_val.to_numpy().flatten()               # True class indices
accuracy = np.mean(val_preds == val_labels)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


index = 8

sample_input = X_val_processed_np[index].reshape(1, -1)  # Add batch dimension
output = model.forward(sample_input)
predicted_class = np.argmax(output)
print("Predicted class:", predicted_class)

X_val = X_val * 255.0 
# Extract the row from the DataFrame as a NumPy array
sample = X_val.iloc[index].values  # Access the data row as a NumPy array

# Reshape to 28x28 for visualization
image = sample.reshape(28, 28)

# Display the image
utils.show_image(image, title=f"Original Input Image at index {index}", subtitle=f"Predicted class: {predicted_class}")