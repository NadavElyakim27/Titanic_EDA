"""
LogisticRegression Model
This module contains the implementation of a LogisticRegression model for classification.
In this module, I demonstrated the use of OOP - classes and inheritance.
"""

import abc
import numpy as np
import matplotlib.pyplot as plt
from src.evaluation import accuracy_clac


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    Just to demonstrate inheritance and OOP.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass


class MybBinaryCrossEntropy(ClassifierLoss):
    """
    BinaryCrossEntropy loss
    """

    def __init__(self):
        self.epsilon = 1e-9

    def loss(self, y_true, y_pred):
        """
        Calculates the Binary-Cross-Entropy-loss
        Args:
            y_true: true label
            y_pred: predicted probabilities
        Returns:
            loss: binary cross entropy
        """
        y1 = y_true * np.log(y_pred + self.epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + self.epsilon)
        loss = -np.mean(y1 + y2)

        return loss


class Classifier:
    """
    Represents a Classifier model.
    Just to demonstrate inheritance and OOP.
    """

    def __init__(
        self, data: dict = None, loss_fn: ClassifierLoss = MybBinaryCrossEntropy()
    ):
        """
        Initializes the Classifier model.
        Args:
            data: dict of all data (X_train, X_test, y_train, y_test)
            loss_fn: the loss function
        """
        # Data
        self.data = data
        self.X_train = data["X_train"].astype(float)
        self.X_test = data["X_test"].astype(float)
        self.y_train = data["y_train"].astype(float)
        self.y_test = data["y_test"].astype(float)
        # Loss
        self.loss_fn = loss_fn


class LogisticRegression(Classifier):
    """
    Logistic Regression implementation from scratch.
    """

    def __init__(
        self,
        data: dict = None,
        loss_fn: ClassifierLoss = MybBinaryCrossEntropy(),
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: str = "l2",
        lambda_reg: float = 0.01,
    ):
        """
        Initializes the Logistic Regression classifier.
        Args:
            data: dict of all data (X_train, X_test, y_train, y_test)
            loss_fn: the loss function
            learning_rate: learning rate of gradient descent
            n_iterations: (int): max iterations for gradient descent
            regularization: 'l1' for (Lasso) 'l2' for Ridge
            lambda_reg: regularization hyperparameter
        """
        super().__init__(data, loss_fn)

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = np.zeros(self.X_train.shape[1])
        self.bias = 0
        self.feature_importances_ = None

    def fit(self, plot: bool = False):
        """
        Fit the logistic regression model to the training data
        Returns:
            train and test losses
        """
        # Losses and accuracy to memory lists
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []

        for _ in range(self.n_iterations):

            # Predict for accuracy
            y_pred_train, logits = self.predict(self.X_train)
            y_pred_test, _ = self.predict(self.X_test)

            # A losses and accuracy to memory lists
            train_losses.append(self.loss_fn.loss(self.y_train, logits))
            test_losses.append(self.loss_fn.loss(self.y_test, y_pred_test))
            train_acc.append(accuracy_clac(self.y_train, y_pred_train))
            test_acc.append(accuracy_clac(self.y_test, y_pred_test))

            # Gradient
            dw = np.dot(self.X_train.T, (logits - self.y_train)) / len(self.y_train)
            db = np.sum(logits - self.y_train) / len(self.y_train)

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        # Save feature importances
        self.feature_importances_ = np.abs(self.weights)

        # Plot if need
        if plot:
            plot_losses_accuracy((train_losses, test_losses), (train_acc, test_acc))

        return (train_losses, test_losses)

    def predict(self, X):
        """
        Predict binary labels for the input data.
        Args:
            X: array-like of shape (n_samples, n_features)
        Returns:s
            y_perd: Array of predicted binary labels
        """

        # Calculate X*T + b
        z = (np.dot(X, self.weights) + self.bias).astype(float)

        # Apply sigmoid function to get probabilities
        predictions = sigmoid(z)

        # Convert probabilities to binary predictions
        y_perd = np.round(predictions).astype(int)

        # Apply regularization
        if self.regularization == "l1":
            self.weights -= self.lambda_reg * np.sign(self.weights)
        elif self.regularization == "l2":
            self.weights -= 2 * self.lambda_reg * self.weights

        return y_perd, predictions


def plot_losses_accuracy(losses, accuracy):
    """
    Plot train and test loss and accuracy after train the model
    Args:
        losses: taple - train loss, test loss
        accuracy: taple - accuracy loss, accuracy loss
    Returns:
        show plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Train loss plot
    axs[0][0].plot(losses[0], label="Training Loss")
    axs[0][0].set_xlabel("Epochs")
    axs[0][0].set_ylabel("Loss")
    axs[0][0].set_title("Training Loss over Epochs")
    # Test loss plot
    axs[0][1].plot(losses[1], label="Test Loss")
    axs[0][1].set_xlabel("Epochs")
    axs[0][1].set_title("Test Loss over Epochs")

    # Train accuracy plot
    axs[1][0].plot(accuracy[0], label="Training accuracy")
    axs[1][0].set_xlabel("Epochs")
    axs[1][0].set_ylabel("accuracy")
    axs[1][0].set_title("Training accuracy over Epochs")
    # Test accuracy plot
    axs[1][1].plot(accuracy[1], label="Test accuracy")
    axs[1][1].set_xlabel("Epochs")
    axs[1][1].set_title("Test accuracy over Epochs")

    plt.show()


def sigmoid(z):
    """
    Apply the sigmoid.
    Args:
        z: Input values
    Returns:
        array: sigmoid function calculation
    """
    return 1 / (1 + np.exp(-z))
