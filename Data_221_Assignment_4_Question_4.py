# Assignment 4 Question 4
# Joseph Krosel

# Loads the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Transforms the data for processing
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = (
    train_test_split(data.data, data.target, test_size=0.20, random_state=42))

# Standardize the input features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Creates the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import accuracy_score

neural_network_model = Sequential()

# Input layer
input_layer = InputLayer(input_shape=(30,))
neural_network_model.add(input_layer)

# Hidden layer
hidden_layer = Dense(3)
neural_network_model.add(hidden_layer)

# Output layer
output_layer = Dense(1, activation='sigmoid')
neural_network_model.add(output_layer)

# Trains the model
neural_network_model.compile(optimizer= 'adam', loss='binary_crossentropy')
neural_network_model.fit(features_train, labels_train, epochs=5)

# Predicts training and testing data
predicted_labels_for_training = neural_network_model.predict(features_train)
predicted_labels_for_testing = neural_network_model.predict(features_test)

# Convert probabilities to 0 or 1
predicted_labels_for_training = (predicted_labels_for_training > 0.5).astype(int)
predicted_labels_for_testing = (predicted_labels_for_testing > 0.5).astype(int)

# Accuracy for training and testing data
accuracy_for_training = accuracy_score(labels_train, predicted_labels_for_training)
print("Accuracy on training data", accuracy_for_training)

accuracy_for_testing = accuracy_score(labels_test, predicted_labels_for_testing)
print("Accuracy on testing data", accuracy_for_testing)

# Question 1: Why feature scaling is necessary for neural networks.
# Answer: Feature scaling ensures stable, fast training and better accuracy because
# the data is now in closer ranges.

# Question 2: what an epoch represents during neural network training.
# Answer: An epoch represents a training cycle that went through a complete pass of the training data set
