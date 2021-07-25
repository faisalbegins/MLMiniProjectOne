import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


def plot_validation_curve(estimator=None, title='Validation Curve', X=None, y=None, param_name=None,
                          parameter_range=None):
    train_score, test_score = validation_curve(estimator,
                                               X, y,
                                               param_name=param_name,
                                               param_range=parameter_range,
                                               cv=5, scoring="accuracy")

    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis=1)
    std_train_score = np.std(train_score, axis=1)

    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis=1)
    std_test_score = np.std(test_score, axis=1)

    # Plot mean accuracy scores for training and testing scores
    plt.plot(parameter_range, mean_train_score, label="Training Score", color='b')
    plt.plot(parameter_range, mean_test_score, label="Cross Validation Score", color='g')

    # Creating the plot
    plt.title(title)
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


def plot_learning_curve(title='Learning Curve', estimator=None, X=None, y=None):
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator,
                                                            X=X, y=y,
                                                            cv=10,
                                                            n_jobs=1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))

    # Calculate training and test mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--',
             label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title(title)
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
