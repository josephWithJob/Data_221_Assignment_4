# Assignment 4, Question 7
# Joseph Krosel

# Loading data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Processing data (normalizing)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshaping the image
x_train = x_train[..., None]
x_test = x_test[..., None]

# Building the CNN model
from tensorflow.keras import layers, models
model = models.Sequential([layers.Input(shape=(28, 28, 1)),
                           layers.Conv2D(16, 3, padding="same", activation="relu"),
                           layers.MaxPool2D(),
                           layers.Conv2D(32, 3, padding="same", activation="relu"),
                           layers.MaxPool2D(),])
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation="softmax"))

# Training the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"] )
history = model.fit(x_train, y_train, validation_split=0.1,
                    epochs=15, batch_size=64)

# Predicting
CNN_predictions = model.predict(x_test, verbose=0)

# Convert probabilities to class labels
CNN_predictions = np.argmax(CNN_predictions, axis=1)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
CNN_confusion_matrix_of_classification = confusion_matrix(y_test, CNN_predictions)

# Output the Confusion Matrix
print("CONFUSION MATRIX")
print(CNN_confusion_matrix_of_classification)

# Output the misclassified images
counter = 0
misclassifiedImages = []
for i in range(len(y_test)):
    if y_test[i] != CNN_predictions[i]:
        counter += 1
        misclassifiedImages.append(i)
        if counter >= 3:
            break

import matplotlib.pyplot as plt
for i in misclassifiedImages:
    text = f'True label: {y_test[i]}. Predicted Label: {CNN_predictions[i]}'
    plt.matshow(x_train[i], cmap='gray')
    plt.text(0, -3, text, bbox=dict(facecolor='black', alpha=0))
    plt.show()

# Question 1: One pattern you observe in the misclassifications
# Answer: Foot wear is a common issue for the model to predict.

# Question 2: One realistic method to improve the CNN performance.
# Answer: Train the model for a greater number of epochs to train the model even more.