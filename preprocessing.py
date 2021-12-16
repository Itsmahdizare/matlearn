import numpy as np

def polynomial(x, degree=1, add_bias_coefs=False):
    """used to calculate the polynomial coefficients of a given array.

    Args:
        x (array): the input array to be calculated.
        degree (int, optional): polynominal degree. Defaults to 1.
        add_bias_coefs (bool, optional): set True if you wan to add bias coefficients to the input array. This will add a column to the input array containing only ones.
        Equivalent to np.ones(x.shape[0]). Defaults to False.

    Returns:
        Array
    """
    if degree > 1:
        for i in range(2, degree+1):
            x = np.hstack((x, x**i))
    if add_bias_coefs:
        x = np.hstack((np.ones((x.shape[0], 1)), x))
    return x


def standardization(x, just_mean=False):
    """Used to standardize the input array. The result will be an array with a mean of zero and a standard deviation of one. 

    Args:
        x (Array): the input array.
        just_mean (bool, optional): If True, then only mean value of the input array will be zero and the standard deviation is not affected. Defaults to False.

    Returns:
        Array: A standard array with mean = 0 and std = 1.
    """
    if just_mean:
        return (x - np.mean(x))
    return (x - np.mean(x)) / (np.std(x))


def normalization(x, mean_norm=False):
    """Used to normalize the input array.

    Args:
        x (Array): the input array to be normalized.
        mean_norm (bool, optional): if True, the mean normalization is applied. Defaults to False.

    Returns:
        Array: A normalized array with mean = 0 and std = 1.
    """
    if mean_norm:
        return ((x - np.mean(x)) / (np.max(x) - np.min(x)))
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def split_train_test(x, y, train_ratio=.8, shuffle=False):
    """Used to split the data into train and test

    Args:
        x (Array): Features matrix. 
        y ([type]): Target matrix.
        train_ratio (float, optional): A float between 0 and 1. Sets the ratio of the training set. Defaults to .8.
        shuffle (bool, optional): If True, shuffles the data before training. Defaults to False.

    Returns:
        Array, Array, Array, Array: x_train, x_test, y_train, y_test
    """
    if shuffle:
        tempArray = np.concatenate((x, y), axis=1)
        np.random.shuffle(tempArray)
        x = tempArray[:, :-1]
        y = tempArray[:, -1].reshape(-1, 1)
        del tempArray
    idx = int(train_ratio * len(x))
    def train(A): return A[0: idx]
    def test(B): return B[idx:]
    return (train(x), test(x), train(y), test(y))