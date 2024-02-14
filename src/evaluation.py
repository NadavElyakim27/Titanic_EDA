import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Utility module for evaluation method and plots
"""

def accuracy_clac(y_true, y_pred):
    """
    Calculate the accuracy score of a classification model.
    Args:
        y_true: pd.Series. True labels.
        y_pred: pd.Series. Predicted labels.
    Returns:
        Accuracy score.
    """
    correct_predictions = sum(y_true == y_pred)
    total_samples = len(y_true)

    accuracy = (correct_predictions / total_samples)*100
    return accuracy


def confusion_matrix1(y_true, y_pred):
    """
    Calculate the confusion matrix of a classification model.
    Args:
        y_true: True labels array.
        y_pred: Predicted labels array.
    Returns:
        Confusion matrix.
    """
    true_positive = sum((true == 1) and (pred == 1) for true, pred in zip(y_true, y_pred))
    true_negative = sum((true != 1) and (pred != 1) for true, pred in zip(y_true, y_pred))
    false_positive = sum((true != 1) and (pred == 1) for true, pred in zip(y_true, y_pred))
    false_negative = sum((true == 1) and (pred != 1) for true, pred in zip(y_true, y_pred))

    confusion_matrix = pd.DataFrame(
        {
            'Predicted Survived': [true_positive, false_positive],
            'Predicted not Survived': [false_negative, true_negative],
        },
        index=['Actual - Survived ', 'Actual Not Survived']
    )

    return confusion_matrix

def plot_result(conf_matrix, accuracy):
    """
    Plot the confusion matrix of a classification model with the accuracy.
    Args:
        conf_matrix: Confusion matrix results.
        accuracy: results.
    Returns:
        plot
    """

    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                annot_kws={"size": 16}
                )
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}')
    plt.show()


def plot_importances(model, X):
    """
    Plot the importances of features from trained model.
    Args:
        model: Trained model
        X: model's dataset
    Returns:
        plot
    """
    # Get feature importances from the trained Random Forest model
    feature_importances = model.feature_importances_

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

    # Display a bar plot of feature importances
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature importances in the model')
    plt.show()
