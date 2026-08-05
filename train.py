"""
Trains the softmax regression model from scratch and saves the learned parameters to a file that pred.py can load.

Should be used just once to produce model_params.npz. 
Based on lab 4, extended to multiple classes.
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


def loss(W, X, T, reg_lambda=0.0):
    """
    Average cross-entropy loss across all examples, plus an L2 penalty on
    the weights (not the bias row) to discourage overfitting.
    T is the one-hot target matrix (shape N x K).
    """
    Y = pred(W, X)
    # small number added so it never takes log(0)
    cross_entropy = np.mean(-np.sum(T * np.log(Y + 1e-12), axis=1))
    l2_penalty = (reg_lambda / 2) * np.sum(W[1:] ** 2)
    return cross_entropy + l2_penalty


def accuracy(W, X, T):
    """
    Fraction of examples where the predicted class matches the true class.
    """
    Y = pred(W, X)
    predicted_class = np.argmax(Y, axis=1)
    true_class = np.argmax(T, axis=1)
    return np.mean(predicted_class == true_class)


def grad(W, X, T, reg_lambda=0.0):
    """
    Gradient of the cross-entropy loss with respect to W, plus the
    derivative of the L2 penalty (not applied to the bias row).
    Vectorized version of the binary gradient.
    """
    Y = pred(W, X)
    data_grad = np.dot(X.T, (Y - T)) / len(T)
    reg_grad = reg_lambda * W
    reg_grad[0] = 0  # never penalize the bias row
    return data_grad + reg_grad


def train(X_train, T_train, X_val=None, T_val=None, alpha=0.5, n_iters=2000, reg_lambda=0.0, plot=True):
    """
    Gradient descent.
    Prints progress and (if plot=True) plots the training curve at the end.
    X_val/T_val are optional -- pass them to also track validation loss/accuracy.
    """
    num_features = X_train.shape[1]
    num_classes = T_train.shape[1]
    W = np.zeros((num_features, num_classes))

    train_losses = []
    val_losses = [] if X_val is not None else None

    for i in range(n_iters):
        W = W - alpha * grad(W, X_train, T_train, reg_lambda)

        train_losses.append(loss(W, X_train, T_train, reg_lambda))
        if val_losses is not None:
            val_losses.append(loss(W, X_val, T_val, reg_lambda))

    if plot:
        plt.figure()
        plt.plot(train_losses, label='train loss')
        if val_losses is not None:
            plt.plot(val_losses, label='val loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.title(f'Training curve (reg_lambda={reg_lambda})')
        plt.savefig('training_curve.png')
        print('Saved training_curve.png')

    msg = f'reg_lambda={reg_lambda}: train accuracy={accuracy(W, X_train, T_train):.4f}'
    if X_val is not None:
        msg += f', val accuracy={accuracy(W, X_val, T_val):.4f}'
    print(msg)

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

    # 'id' is only for grouping respondents during train/val/test splitting
    # and cross-validation elsewhere -- not a real feature. Raw Q7/Q8/Q9 are
    # only here so we can recompute capping bounds below; the model uses the
    # *_capped versions, not these raw columns.
    feature_cols = [c for c in train_df.columns if c not in ('Label', 'id', 'Q7', 'Q8', 'Q9')]
    classes = sorted(train_df['Label'].unique())

    # reg_lambda was already picked using val (see git history / notebook),
    # so there's no more reason to hold val out of the final fit -- fold it
    # into the training set for ~21% more data. test_raw.csv is still a
    # fully independent set never touched here.
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Recompute Q7-9 capping bounds from train+val combined, since that's
    # now the population the final model is actually fit on -- the bounds
    # baked into the CSVs by the cleaning notebook were train-only, computed
    # before val was folded back in here.
    q7_lo, q7_hi = -20, 45  # fixed by choice, not data-derived
    q8_hi = full_train_df['Q8'].quantile(0.99)
    q9_hi = full_train_df['Q9'].quantile(0.99)
    full_train_df['Q7_capped'] = full_train_df['Q7'].clip(q7_lo, q7_hi)
    full_train_df['Q8_capped'] = full_train_df['Q8'].clip(upper=q8_hi)
    full_train_df['Q9_capped'] = full_train_df['Q9'].clip(upper=q9_hi)
    print(f'Recomputed capping bounds (train+val, n={len(full_train_df)}): '
          f'q7=[{q7_lo}, {q7_hi}], q8_hi={q8_hi!r}, q9_hi={q9_hi!r}')

    # standardize using the full (train+val) mean/std only
    mean = full_train_df[feature_cols].mean()
    std = full_train_df[feature_cols].std()

    X_train = ((full_train_df[feature_cols] - mean) / std).to_numpy()

    # add a column of 1s for the bias/intercept
    X_train = np.hstack([np.ones((len(X_train), 1)), X_train])

    T_train = one_hot(full_train_df['Label'].tolist(), classes)

    reg_lambda = 0.1
    W = train(X_train, T_train, alpha=0.5, n_iters=500, reg_lambda=reg_lambda)

    # save everything pred.py will need:  weights,  mean/std used
    # for standardizing, and the class order (so argmax index -> city name)
    np.savez('model_params.npz',
             W=W,
             mean=mean.to_numpy(),
             std=std.to_numpy(),
             feature_cols=feature_cols,
             classes=classes,
             reg_lambda=reg_lambda)
    print('Saved model_params.npz')
