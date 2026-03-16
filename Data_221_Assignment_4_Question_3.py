# Assignment 4 Question 3
# Joseph Krosel

# Loads the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Transforms the data for processing
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = (
    train_test_split(data.data, data.target, test_size=0.20, random_state=42))

# Creates the model and trains the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Uses max_depth = 10
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=10)
decision_tree_classifier.fit(features_train, labels_train)

# Evaluates the model on training data
predicted_labels_for_training = decision_tree_classifier.predict(features_train)
accuracy_for_training = accuracy_score(labels_train, predicted_labels_for_training)
print("Accuracy on training data", accuracy_for_training)

# Evaluates the model on testing data
predicted_labels_for_testing = decision_tree_classifier.predict(features_test)
accuracy_for_testing = accuracy_score(labels_test, predicted_labels_for_testing)
print("Accuracy on testing data", accuracy_for_testing)

# Finds the important features
model_important_features = decision_tree_classifier.feature_importances_
top_five_important_features = sorted(model_important_features, reverse=True)[:5]

# Prints the important features
print("TOP FIVE IMPORTANT FEATURES")
for i in top_five_important_features:
    print(i)

# Question 1: How controlling model complexity affects overfitting
# Answer: A models complexity that is too flexible and maps itself to all the features
# instead of catching general patterns, causes overfitting.

# Question 2: How feature importance contributes to the interpretability of decision trees.
# Answer: The feature importance directly correlates to the interpretability of the decision tree
# because these features are what partition the data and help make the decisions the model makes.