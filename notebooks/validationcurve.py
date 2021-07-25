import numpy as np
import matplotlib.pyplot as plt
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
    plt.xlabel("y")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

