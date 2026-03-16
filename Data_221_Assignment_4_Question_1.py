# Assignment 4 Question 1
# Joseph Krosel

# Loads the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Constructs feature matrix x and target y
feature_matrix = data.data
target_matrix = data.target

# Reports the shape
print("The shape of the feature matrix is a rectangle")
print("The shape of the target matrix is a line")

# Reports the number of samples per class
print("Size of feature matrix", len(feature_matrix) * len(feature_matrix[1]))
print("Size of target matrix", len(target_matrix))

# Question 1: Whether the dataset is balanced or imbalanced.
# Answer: The data set is imbalanced because both the target matrix
# and the feature matrix have differing numbers of samples.

# Question 2: Why class balance is an important consideration for classification models
# Answer: When data is skewed, a models accuracy is inaccurate because the model could have
# just picked the most common classification other than actually predicting and analyzing.