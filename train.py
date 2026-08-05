"""
Trains the softmax regression model from scratch and saves the learned parameters to a file that pred.py can load.

Should be used just once to produce model_params.npz. 
Same idea as lab04, just extended from binary (sigmoid) to multi-class (softmax).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(z):
    """
    Turn a matrix of raw scores z (shape N x K) into probabilities that add up to 1 across each row.
    """
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def pred(W, X):
    """
    Given weights W and inputs X, return the predicted probabilities for each class.
    """
    z = np.dot(X, W)
    return softmax(z)


def loss(W, X, T):
    """
    Average cross-entropy loss across all examples.
    T is the one-hot target matrix (shape N x K).
    """
    Y = pred(W, X)
    # small number added so it never takes log(0)
    return np.mean(-np.sum(T * np.log(Y + 1e-12), axis=1))


def accuracy(W, X, T):
    """
    Fraction of examples where the predicted class matches the true class.
    """
    Y = pred(W, X)
    predicted_class = np.argmax(Y, axis=1)
    true_class = np.argmax(T, axis=1)
    return np.mean(predicted_class == true_class)


def grad(W, X, T):
    """
    Gradient of the cross-entropy loss with respect to W.
    Same formula as lab04's binary case, just using the full Y-T matrix instead of a single y-t column.
    """
    Y = pred(W, X)
    return np.dot(X.T, (Y - T)) / len(T)


def train(X_train, T_train, X_val, T_val, alpha=0.5, n_iters=2000):
    """
    Gradient descent, same loop as lab04's solve_via_gradient_descent. 
    Prints progress and plots the training curve at the end.
    """
    num_features = X_train.shape[1]
    num_classes = T_train.shape[1]
    W = np.zeros((num_features, num_classes))

    train_losses = []
    val_losses = []

    for i in range(n_iters):
        W = W - alpha * grad(W, X_train, T_train)

        train_losses.append(loss(W, X_train, T_train))
        val_losses.append(loss(W, X_val, T_val))

    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Training curve')
    plt.savefig('training_curve.png')
    print('Saved training_curve.png')

    print('Final train accuracy:', accuracy(W, X_train, T_train))
    print('Final val accuracy:  ', accuracy(W, X_val, T_val))

    return W


def one_hot(labels, classes):
    """
    Turn a list of city names into a one-hot matrix, e.g.
    'Dubai' -> [1, 0, 0, 0] if classes = ['Dubai', 'NYC', 'Paris', 'Rio']
    """
    T = np.zeros((len(labels), len(classes)))
    for i, label in enumerate(labels):
        class_index = classes.index(label)
        T[i, class_index] = 1
    return T


if __name__ == '__main__':
    train_df = pd.read_csv('train_features.csv')
    val_df = pd.read_csv('val_features.csv')

    feature_cols = [c for c in train_df.columns if c != 'Label']
    classes = sorted(train_df['Label'].unique())

    # standardize using train mean/std only
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()

    X_train = ((train_df[feature_cols] - mean) / std).to_numpy()
    X_val = ((val_df[feature_cols] - mean) / std).to_numpy()

    # add a column of 1s for the bias/intercept
    X_train = np.hstack([np.ones((len(X_train), 1)), X_train])
    X_val = np.hstack([np.ones((len(X_val), 1)), X_val])

    T_train = one_hot(train_df['Label'].tolist(), classes)
    T_val = one_hot(val_df['Label'].tolist(), classes)

    W = train(X_train, T_train, X_val, T_val, alpha=0.5, n_iters=500)

    # save everything pred.py will need:  weights,  mean/std used
    # for standardizing, and the class order (so argmax index -> city name)
    np.savez('model_params.npz',
             W=W,
             mean=mean.to_numpy(),
             std=std.to_numpy(),
             feature_cols=feature_cols,
             classes=classes)
    print('Saved model_params.npz')
