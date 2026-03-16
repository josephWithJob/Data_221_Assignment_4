# Assignment 4 Question 2
# Joseph Krosel

# Loads the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Transforms the data for processing
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = (
    train_test_split(data.data, data.target, test_size=0.20, random_state=42))

# Creates the model and trains it
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(features_train, labels_train)

# Evaluates the model on training data
predicted_labels_for_training = decision_tree_classifier.predict(features_train)
accuracy_for_training = accuracy_score(labels_train, predicted_labels_for_training)
print("Accuracy on training data", accuracy_for_training)

# Evaluates the model on testing data
predicted_labels_for_testing = decision_tree_classifier.predict(features_test)
accuracy_for_testing = accuracy_score(labels_test, predicted_labels_for_testing)
print("Accuracy on testing data", accuracy_for_testing)

# Question 1: What entropy represents in the context of decision trees
# Answer: Entropy represent the level of uncertainty at which a splitting of a partition data.
# This is used to guide the algorithm to split nodes in a way that maximizes information gained.

# Question 2: Whether the observed results suggest overfitting or good generalization
# Answer: It would suggest a good generalization because the accuracy of the model is quite good at a level of 0.968%.
