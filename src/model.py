"""
LogisticRegression Model

This module contains the implementation of a LogisticRegression model for classification.
"""

import numpy as np

class LogisticRegression:
    """
    Logistic Regression implementation from scratch.
    """

    def __init__(self,
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

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data
        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)
        """

        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            # Calculate X*T + b
            z = np.dot(X, self.weights) + self.bias

            # Apply sigmoid function to get probabilities
            probabilities = self.sigmoid(z)

            # Gradient
            gradient_weights = np.dot(X.T, (probabilities - y)) / len(y)
            gradient_bias = np.sum(probabilities - y) / len(y)

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

        self.feature_importances_ =  np.abs(self.weights) 

    def predict(self, X):
        """
        Predict binary labels for the input data.
        Args:
            X: array-like of shape (n_samples, n_features)
        Returns:
            binary_predictions: Array of predicted binary labels
        """
        
        # Calculate X*T + b
        z = np.dot(X, self.weights) + self.bias

        # Apply sigmoid function to get probabilities
        predictions = self.sigmoid(z)

        # Convert probabilities to binary predictions
        binary_predictions = np.round(predictions).astype(int)

        # Apply regularization
        if self.regularization == 'l1':
            self.weights -= self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            self.weights -= 2 * self.lambda_reg * self.weights

        
        return binary_predictions
