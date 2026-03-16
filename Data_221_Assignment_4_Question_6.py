# Assignment 4, Question 6
# Joseph Krosel

# Loading data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

# Question 1: Why CNNs are generally preferred over fully connected networks for image data
# Answer: CNNs are designed to handle image data where the surrounding pixels are necessary to
# decide an outcome.

# Question 2: What the convolution layer is learning in this task
# Answer: The convolution layer is learning to look a kernels and find the correct patterns that
# correlate to the labels