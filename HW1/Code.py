# %%
# Import everything we need
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# %%
# To understand this dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('./Resources/2024_energy_efficiency_data.csv')

# 1. Initial inspection
print("Shape of dataset:", data.shape)
print("Data types and first few rows:\n", data.dtypes, data.head())

# 2. Check for missing data
missing_data = data.isnull().sum()
print("Missing data in each column:\n", missing_data)

# 3. Summary statistics for numerical features
print("Summary statistics for numerical features:\n", data.describe())

# 4. Unique values in categorical features
categorical_cols = ['Orientation', 'Glazing Area Distribution']
for col in categorical_cols:
    print(f"Unique values in {col}:", data[col].unique())
    print(f"Value counts in {col}:\n", data[col].value_counts())

# 5. Visualizing distributions
data.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.show()

# Bar plot for categorical features
for col in categorical_cols:
    data[col].value_counts().plot(kind='bar', title=f'Distribution of {col}')
    plt.show()

# 6. Correlation matrix
corr_matrix = data.corr()
print("Correlation matrix:\n", corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.show()

# 7. Target distribution (Heating Load, Cooling Load)
print("Heating Load Distribution:\n", data['Heating Load'].describe())
data['Heating Load'].plot(kind='hist', title='Heating Load Distribution')
plt.show()

print("Cooling Load Distribution:\n", data['Cooling Load'].describe())
data['Cooling Load'].plot(kind='hist', title='Cooling Load Distribution')
plt.show()

# 8. One-hot encode categorical features
data_encoded = pd.get_dummies(data, columns=['Orientation', 'Glazing Area Distribution'], prefix=['Orientation', 'GlazingAreaDist'])
print("Data after one-hot encoding:\n", data_encoded.head())


# %%
# 1. Load data and preprocessing data
data = pd.read_csv('./Resources/2024_energy_efficiency_data.csv')

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# One hot 
data = pd.get_dummies(data, columns=["Orientation", "Glazing Area Distribution"], prefix=["Orientation", "GlazingAreaDist"], dtype=int)

# Split to X and Y
X = data.drop(columns=["Heating Load", "Cooling Load"]) 
Y = data[["Heating Load", "Cooling Load"]]

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


# %%
# Create functions to train model
# Initialize weights and biases
def initialize_weights(input_dim, hidden_dim, output_dim):
    # Randomly initialize weights and biases for each layer
    W1 = np.random.uniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim), (input_dim, hidden_dim))
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.uniform(-1/np.sqrt(hidden_dim), 1/np.sqrt(hidden_dim), (hidden_dim, output_dim))
    b2 = np.zeros((1, output_dim))
    
    return W1, b1, W2, b2

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative (used in backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# ReLU derivative (for backpropagation)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1  # Linear combination for hidden layer
    A1 = relu(Z1)            # Apply ReLU activation in hidden layer
    Z2 = np.dot(A1, W2) + b2  # Linear combination for output layer
    A2 = Z2                   # No activation for regression (or sigmoid for classification)
    return Z1, A1, Z2, A2

# Backward propagation for regression
def backward_propagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]  # Number of samples
    dZ2 = A2 - Y  # Derivative of cost function (Mean Squared Error for regression)
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)  # Derivative for ReLU activation
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0)
    
    return dW1, db1, dW2, db2


# Update weights and biases
def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2



# %%
# Train the model
def train(X, Y, input_dim, hidden_dim, output_dim, epochs, learning_rate):
    # Initialize weights
    W1, b1, W2, b2 = initialize_weights(input_dim, hidden_dim, output_dim)

    for i in range(epochs):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        
        # Calculate loss (MSE for regression)
        loss = np.mean((A2 - Y) ** 2)
        print(f'Epoch {i+1}/{epochs}, Loss: {loss}')

        # Backward propagation
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W2)
        
        # Update weights
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    
    return W1, b1, W2, b2

# %%
# Train Processing
input_dim = X_train.shape[1]  # Number of features
hidden_dim = 8  # Example hidden layer size (you can adjust this)
output_dim = 1  # Predicting one continuous value (Heating Load)
epochs = 1000  # You can adjust the number of epochs
learning_rate = 0.005  # Learning rate for gradient descent

# Train the model
W1, b1, W2, b2 = train(X_train, Y_train['Heating Load'].values.reshape(-1, 1),
                       input_dim, hidden_dim, output_dim, epochs, learning_rate)



