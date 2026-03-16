# Assignment 4 Question 5
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

# Creates the model neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

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
neural_network_predicted_labels_for_testing = neural_network_model.predict(features_test)

# Convert probabilities to 0 or 1
neural_network_predicted_labels_for_testing = (neural_network_predicted_labels_for_testing > 0.5).astype(int)

# Creates the decision tree
# Transforms the data for processing
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = (
    train_test_split(data.data, data.target, test_size=0.20, random_state=42))

# Creates the model
from sklearn.tree import DecisionTreeClassifier

decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=10)
decision_tree_classifier.fit(features_train, labels_train)

# Evaluates the model on testing data
decision_tree_predicted_labels_for_testing = decision_tree_classifier.predict(features_test)


# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
decision_tree_confusion_matrix_of_classification = confusion_matrix(labels_test, decision_tree_predicted_labels_for_testing)
print("DECISION TREE MODEL")
print(decision_tree_confusion_matrix_of_classification)

print("")

neural_network_tree_confusion_matrix_of_classification = confusion_matrix(labels_test, neural_network_predicted_labels_for_testing)
print("NEURAL NETWORK MODEL")
print(neural_network_tree_confusion_matrix_of_classification)

# Question 1: Which model you would prefer for this task.
# Answer: I would prefer to use the decision tree model because when testing the
# neural network model multiple times. The data was either super inaccurate or a little less
# accurate than the decision tree model.

# Question 2: One advantage and one limitation of each model.
# Answer: Decision trees are hardly effected by outliers, but are prone to overfitting.
# Neural networks process in parallel allowing them to process large amounts of data
# quickly, but are difficult to interpret because they are black box's