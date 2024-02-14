"""
LogisticRegression Model

This module contains the implementation of a LogisticRegression model for classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.evaluation import accuracy_clac

class LogisticRegression:
    """
    Logistic Regression implementation from scratch.
    """

    def __init__(self,
                  data: dict = None,
                  learning_rate: float=0.01,
                  n_iterations: int=1000,
                  regularization: str='l2',
                  lambda_reg: float=0.01
                ):
        """
        Initializion.
        Args:
            learning_rate: learning rate of gradient descent
            n_iterations (int): max iterations for gradient descent
        """
        # Data
        self.data = data
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        # Param
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.feature_importances_ = None
    
    def sigmoid(self, z):
        """
        Apply the sigmoid.
        Args:
            z: Input values
        Returns:
            array: sigmoid function calculation
        """

        return 1 / (1 + np.exp(-z))
    
    def loss_clac(self, y_true, y_pred):
        """
        Compute binary cross entropy loss
        Args:
            y_true: true label
            y_pred: predicted probabilities
        Returns:
            loss: binary cross entropy
        """        

        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
        loss = -np.mean(y1 + y2)
        
        return loss
        
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

        # Initialize weights and bias
        self.weights = np.zeros(self.X_train.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            # Calculate X*T + b
            z = np.dot(self.X_train, self.weights) + self.bias

            # Apply sigmoid function to get probabilities
            y_perd = self.sigmoid(z)

            # Predict for accuracy
            y_train_pred = self.predict(self.X_train)
            y_pred_test = self.predict(self.X_test)
            # A losses and accuracy to memory lists
            train_losses.append(self.loss_clac(self.y_train, y_perd))
            test_losses.append(self.loss_clac(self.y_test, y_pred_test))
            train_acc.append(accuracy_clac(self.y_train, y_train_pred))
            test_acc.append(accuracy_clac(self.y_test, y_pred_test))

            # Gradient
            dw = np.dot(self.X_train.T, (y_perd - self.y_train)) / len(self.y_train)
            db = np.sum(y_perd - self.y_train) / len(self.y_train)

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        # Save feature importances
        self.feature_importances_ =  np.abs(self.weights)

        # Plot if need
        if plot: plot_losses_accuracy((train_losses, test_losses), (train_acc,test_acc))

        return (train_losses, test_losses)
    
    def predict(self, X):
        """
        Predict binary labels for the input data.
        Args:
            X: array-like of shape (n_samples, n_features)
        Returns:
            y_perd: Array of predicted binary labels
        """
        
        # Calculate X*T + b
        z = np.dot(X, self.weights) + self.bias

        # Apply sigmoid function to get probabilities
        predictions = self.sigmoid(z)

        # Convert probabilities to binary predictions
        y_perd = np.round(predictions).astype(int)

        # Apply regularization
        if self.regularization == 'l1':
            self.weights -= self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            self.weights -= 2 * self.lambda_reg * self.weights

        return y_perd
    

def plot_losses_accuracy(losses, accuracy):
    """
    Plot train and test loss and accuracy after train the model
    Args:
        losses: taple - train loss, test loss
        accuracy: taple - accuracy loss, accuracy loss
    Returns:
        plot
    """
    fig, axs = plt.subplots(2, 2, figsize = (14,12))

    # Train loss plot
    axs[0][0].plot(losses[0], label='Training Loss')
    axs[0][0].set_xlabel('Epochs')
    axs[0][0].set_ylabel('Loss')
    axs[0][0].set_title('Training Loss over Epochs')
    # Test loss plot
    axs[0][1].plot(losses[1], label='Test Loss')
    axs[0][1].set_xlabel('Epochs')
    axs[0][1].set_title('Test Loss over Epochs')

    # Train accuracy plot
    axs[1][0].plot(accuracy[0], label='Training accuracy')
    axs[1][0].set_xlabel('Epochs')
    axs[1][0].set_ylabel('accuracy')
    axs[1][0].set_title('Training accuracy over Epochs')
    # Test accuracy plot
    axs[1][1].plot(accuracy[1], label='Test accuracy')
    axs[1][1].set_xlabel('Epochs')
    axs[1][1].set_title('Test accuracy over Epochs')

    plt.show()
