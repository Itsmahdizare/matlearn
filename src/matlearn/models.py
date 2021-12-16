import numpy as np
from matlearn.evaluation import *
from matlearn._errors import NotTrainingError
from matlearn.preprocessing import split_train_test,polynomial
#########################################################################################################


def activation_function(z, type, derivative=False):
    """Applies activation function to a given input Array.

    Args:
        z (Array): Linear equation of x and w
        type (string): Used to determine the activation function type.
        derivative (bool, optional): If True, the derivative version of the each activation function is applied. Defaults to False.

    Raises:
        ValueError: If an invalid value is given as "type".

    Returns:
        Array: non linear equation of x and w.
    """
    type = type.lower()

    if type == 'relu':
        if derivative:
            z[z <= 0] = 0
            z[z > 0] = 1
            return z
        return np.maximum(0, z)

    elif type == 'leakyrelu':
        if derivative:
            z[z <= 0] = .001
            z[z > 0] = 1
            return z
        z[z <= 0] = .001 * z
        z[z > 0] = z
        return z

    elif type == 'sigmoid':
        if derivative:
            s = activation_function(z, 'sigmoid')
            return s * (1 - s)
        return 1 / (1 + np.exp(-z))

    elif type == 'softmax':
        if derivative:
            s = activation_function(z, 'softmax').reshape(-1, 1)
            return np.diagflat(s) - np.dot(s, s.T)
        # thanks https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    elif type == 'tanh':
        if derivative:
            return 1 - (np.tanh(z) ** 2)
        return np.tanh(z)

    elif type == 'linear':
        return z
    else:
        raise ValueError(f'{type} not supported.')

#################################
class Optimization:

    cost_hist = list()

    def __init__(self):
        pass

    def _check_params(self):

        if self.lr < .0000000001:
            raise ValueError(f'Learning rate must be greater than {self.lr}')
        if self.n_iter_per_epoch < 1:
            raise ValueError(
                'n_iter_per_epoch must be equal or greater than 1.')

    def _set_target(self, y):
        size = np.unique(y).size
        if isinstance(self, LinearRegressor):
            self._app_loss = 'mse'
            self._target_type = 'continuous'
            self._multiclass = False
        elif isinstance(self, LogisticRegressor):
            self._target_type = 'discrete'
            if size == 2:
                self._app_loss = 'binary_cross_entropy'
                self._multiclass = False
            elif size > 2:
                self._app_loss = 'cross_entropy'
                self._multiclass = True

    @staticmethod
    def _weight_grad(x, y, p): return 1/len(x) * np.dot(x.T, (p-y))

    @staticmethod
    def _bias_grad(y, p): return np.average(p - y)

    def GD(self, x, y):
        '''Gradient descent algorithm.
        '''
        self._check_params()
        self._set_target(y)
        if self._target_type == 'discrete':
            self._target_unique_values = np.unique(y)
        self._trained = True
        epoch = 1
        counter = 0
        if y.ndim == 1:
            y = y.reshape(-1,1)

        self.weight = np.random.randn(x.shape[1], 1)
        if self._multiclass:
            self.weight = np.random.randn(x.shape[1], y.shape[1])

        if self.train_bias:
            self.bias = np.ones((self.weight.shape[1],))

        # starting the algorithm
        while epoch != self.maxEpoch+1:

            if self.n_reports > 0:
                print(f'''\n\n **** EPOCH {epoch} ****\n''')

            iterNum = 1
            list_idx = np.array_split(
                np.random.permutation(len(x)), self.n_iter_per_epoch)

            for idx in list_idx:

                xit_train, yit_train = x[idx], y[idx]

                if self.use_validation:
                    xit_train, xit_val, yit_train, yit_val = split_train_test(
                        x[idx], y[idx], shuffle=False, train_ratio=1 - (self.val_ratio))
                    if self._target_type == 'discrete':
                        p_val = self.predict_proba(xit_val)
                    else:
                        p_val = self.predict(xit_val)

                if self._target_type == 'discrete':
                    p_train = self.predict_proba(xit_train)
                else:
                    p_train = self.predict(xit_train)

                if self.train_bias:
                    self.bias = (self.bias) - (self.lr *
                                               Optimization._bias_grad(yit_train, p_train))
                self.weight = self.weight - \
                    (self.lr * Optimization._weight_grad(xit_train, yit_train, p_train))

                dl_train, rl_train = Loss._use_app_loss(self._app_loss)(
                    p_train, yit_train), regularization(self.weight, type=self.regulariz_type, alpha=self.alpha)

                if self.use_validation:
                    dl_val, rl_val = Loss._use_app_loss(self._app_loss)(
                        p_val, yit_val), regularization(self.weight, type=self.regulariz_type, alpha=self.alpha)
                Optimization.cost_hist.append(dl_train + rl_train)

                if self.n_reports > 0:
                    if self.use_validation:
                        print('iteration {:3d}: training loss = {:.2f}  |   validation loss = {:.2f}'.format(
                            iterNum, Optimization.cost_hist[-1], dl_val + rl_val ))
                    else:
                        print('iter {:3d}: training loss = {:.2f}'.format(
                            iterNum, Optimization.cost_hist[-1]))
                iterNum += 1

                try:
                    if abs(Optimization.cost_hist[-1] - Optimization.cost_hist[-2]) < self.converLim:
                        counter += 1
                    if counter == self.n_converLim:
                        print('End of the algorithm at iteration number {}.\nThe differences in costs is less than {}'.format(
                            epoch, self.converLim))
                        return
                except IndexError:
                    pass
            self.n_reports -= 1
            epoch += 1
##########################################################



class _BaseModel(Optimization):

    def __init__(self, regulariz_type, alpha, lr, train_bias, n_reports, maxEpoch, converLim, n_converLim, n_iter_per_epoch, use_validation, val_ratio):

        self.regulariz_type = regulariz_type
        self.alpha = alpha
        self.lr = lr
        self.train_bias = train_bias
        self.n_reports = n_reports
        self.maxEpoch = maxEpoch
        self.converLim = converLim
        self.n_converLim = n_converLim
        self.n_iter_per_epoch = n_iter_per_epoch
        self.use_validation = use_validation
        self.val_ratio = val_ratio
        self._trained = False
        self.bias = 0
        self.weight = 0

    @classmethod
    def score(cls, x, w=None, bias=None, degree=1):
        if degree > 1:
            x = polynomial(x, degree=degree, add_bias_coefs=False)
        return np.dot(x, w) + bias

    def train(self, x, y):
        """Trains the model. Currently all models will train by the gradient descent algorithm.

        Args:
            x (Array): Features matrix.
            y (Array): Target matrix.
        """
        return self.GD(x=x, y=y)


class LinearRegressor(_BaseModel):
    """Linear regression model.

    Args:
        regulariz_type (string, optional): used to specify the type of regularization. Only [L1,L2] are supported. Defaults to L2.
        alpha ([Float, Integer], optional): Alpha hyperparameter. Used to Adjust the importance of the penalty. Defaults to .00001
        lr ([Float, Integer], optional): Set learning rate hyperparameter for gradient descent. Defaults to .0001
        train_bias (bool, optional): Whether to train the bias or not. Defaults to True.
        n_reports (int, optional): Number of reports of training the data. Prints iteration numbers, training loss, and validation loss (if use_validation is True). Set it to 0 to disable printing reports. The maximum number of reports is maxEpoch * n_iter_per_epoch. Defaults to 0
        maxEpoch (int, optional): Maximum number of epoch. Defaults to 100.
        converLim (int, optional): Used to set a threshold for convergence. if differences in loss functions, during training, is less than ConverLim, the model will stop training. Defaults to .001
        n_converLim (int, optional): Number of convergence limits. For Example, if it is set to 5, then  differences in loss functions must be Five times less than ConverLim for five times. Otherwise, the model will continue training. Defaults to 1.
        n_iter_per_epoch (int, optional): Number of interations per epoch. If 0, full batch training is used. If it is set to a value of greater than 1, stochastic batches are used, depending on the number of iterations. For example, is set to 3, then three stochastic batches of data will be used for each epoch. Defaults to 1.
        use_validation (bool, optional): If True, the model will be validated to a training and a validation set. Defaults to True.
        val_ratio (Float, optional): A float number between 0 and 1. Used to set the ratio of validation set. Only use it if use_validation is set to True. Defaults to .1
    """

    def __init__(self, regulariz_type='L2', alpha=.00001, lr=.0001, train_bias=True, n_reports=0, maxEpoch=100, converLim=.001, n_converLim=1, n_iter_per_epoch=1, use_validation=True, val_ratio=.1):
        super().__init__(regulariz_type=regulariz_type, alpha=alpha, lr=lr, train_bias=train_bias, n_reports=n_reports,
                         maxEpoch=maxEpoch, converLim=converLim, n_converLim=n_converLim, n_iter_per_epoch=n_iter_per_epoch, use_validation = use_validation, val_ratio = val_ratio)

    def predict(self, xp):
        """Predicts the values for the given test set, based on model bias and weight. Only works correctly if the model is already trained.

        Args:
            xp (Array): test matrix to be predicted.

        Raises:
            NotTrainingError: If the model is not trained yet.

        Returns:
            Array: predicted values.
        """
        if not self._trained:
            raise NotTrainingError('please train the model before predict.')
        return activation_function(super().score(x=xp, w=self.weight, bias=self.bias), type='linear')


class LogisticRegressor(_BaseModel):
    """Logistic regression model.

    Args:
        regulariz_type (string, optional): used to specify the type of regularization. Only [L1,L2] are supported. Defaults to L2.
        alpha ([Float, Integer], optional): Alpha hyperparameter. Used to Adjust the importance of the penalty. Defaults to .00001
        lr ([Float, Integer], optional): Set learning rate hyperparameter for gradient descent. Defaults to .0001
        train_bias (bool, optional): Whether to train the bias or not. Defaults to True.
        n_reports (int, optional): Number of reports of training the data. Prints iteration numbers, training loss, and validation loss (if use_validation is True). Set it to 0 to disable printing reports. The maximum number of reports is maxEpoch * n_iter_per_epoch. Defaults to 0
        maxEpoch (int, optional): Maximum number of epoch. Defaults to 100.
        converLim (int, optional): Used to set a threshold for convergence. if differences in loss functions, during training, is less than ConverLim, the model will stop training. Defaults to .001
        n_converLim (int, optional): Number of convergence limits. For Example, if it is set to 5, then  differences in loss functions must be Five times less than ConverLim for five times. Otherwise, the model will continue training. Defaults to 1.
        n_iter_per_epoch (int, optional): Number of interations per epoch. If 0, full batch training is used. If it is set to a value of greater than 1, stochastic batches are used, depending on the number of iterations. For example, is set to 3, then three stochastic batches of data will be used for each epoch. Defaults to 1.
        use_validation (bool, optional): If True, the model will be validated to a training and a validation set. Defaults to True.
        val_ratio (Float, optional): A float number between 0 and 1. Used to set the ratio of validation set. Only use it if use_validation is set to True. Defaults to .1
        activ (string, optional): type of activation function. [sigmoid, softmax] are supported. sigmoid is recommended for binary classification and softmax for multi-class classification. Defaults to softmax.
    """

    def __init__(self, regulariz_type='L2', alpha=.00001, train_bias=True, lr=.001, n_reports=0, maxEpoch=100, converLim=.001, n_converLim=1, n_iter_per_epoch=1, use_validation = True, val_ratio = .1, activ='softmax'):
        super().__init__(regulariz_type=regulariz_type, alpha=alpha, train_bias=train_bias, lr=lr, n_reports=n_reports,
                         maxEpoch=maxEpoch, converLim=converLim, n_converLim=n_converLim, n_iter_per_epoch=n_iter_per_epoch, use_validation = use_validation, val_ratio = val_ratio)
        self._activ = activ

    def predict_proba(self, xp):
        """Predicts the probability of values for the given test set, based on model bias and weight. Only works correctly if the model is already trained.

        Args:
            xp (Array): test matrix to be predicted.

        Returns:
            Array: predicted values.
        """
        return activation_function(super().score(
            x=xp, w=self.weight, bias=self.bias), type=self._activ)

    def predict(self, xp):
        """Predicts the values for the given test set, based on model bias and weight. Only works correctly if the model is already trained.

        Args:
            xp (Array): test matrix to be predicted.

        Raises:
            NotTrainingError: If the model is not trained yet.

        Returns:
            Array: predicted values.
        """
        
        if not self._trained:
            raise NotTrainingError('please train the model before predict.')

        p = self.predict_proba(xp)
        if self._multiclass:
            return np.argmax(p, axis=1)
        return np.where(p >= .5, self._target_unique_values.max(), self._target_unique_values.min())