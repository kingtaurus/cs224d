from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix
from numpy import *

class SoftmaxRegression(NNBase):
    """
    Dummy example, to show how to implement a network.
    This implements softmax regression, trained by SGD.
    """

    def __init__(self, wv, dims=[100, 5],
                 reg=0.1, alpha=0.001,
                 rseed=10):
        """
        Set up classifier: parameters, hyperparameters
        """
        ##
        # Store hyperparameters
        self.lreg = reg # regularization
        self.alpha = alpha # default learning rate
        self.nclass = dims[1] # number of output classes

        ##
        # NNBase stores parameters in a special format
        # for efficiency reasons, and to allow the code
        # to automatically implement gradient checks
        # and training algorithms, independent of the
        # specific model architecture
        # To initialize, give shapes as if to np.array((m,n))
        param_dims = dict(W = (dims[1], dims[0]), # 5x100 matrix
                          b = (dims[1])) # column vector
        # These parameters have sparse gradients,
        # which is *much* more efficient if only a row
        # at a time gets updated (e.g. word representations)
        param_dims_sparse = dict(L=wv.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        ##
        # Now we can access the parameters using
        # self.params.<name> for normal parameters
        # self.sparams.<name> for params with sparse gradients
        # and get access to normal NumPy arrays
        self.sparams.L = wv.copy() # store own representations
        self.params.W = random_weight_matrix(*self.params.W.shape)
        # self.params.b1 = zeros((self.nclass,1)) # done automatically!

    def _acc_grads(self, idx, label):
        """
        Accumulate gradients from a training example.
        """
        ##
        # Forward propagation
        x = self.sparams.L[idx] # extract representation
        p = softmax(self.params.W.dot(x) + self.params.b)

        ##
        # Compute gradients w.r.t cross-entropy loss
        y = make_onehot(label, len(p))
        delta = p - y
        # dJ/dW, dJ/db1
        self.grads.W += outer(delta, x) + self.lreg * self.params.W
        self.grads.b += delta
        # dJ/dL, sparse update: use sgrads
        # this stores an update to the row L[idx]
        self.sgrads.L[idx] = self.params.W.T.dot(delta)
        # note that the syntax is overloaded here; L[idx] =
        # works like +=, so if you update the same index
        # twice, it'll store *BOTH* updates. For example:
        # self.sgrads.L[idx] = ones(50)
        # self.sgrads.L[idx] = ones(50)
        # will add -2*alpha to that row when gradients are applied!

        ##
        # We don't need to do the update ourself, as NNBase
        # calls that during training. See NNBase.train_sgd
        # in nn/base.py to see how this is done, if interested.
        ##

    def compute_loss(self, idx, label):
        """
        Compute the cost function for a single example.
        """
        ##
        # Forward propagation
        x = self.sparams.L[idx]
        p = softmax(self.params.W.dot(x) + self.params.b)
        J = -1*log(p[label]) # cross-entropy loss
        Jreg = (self.lreg / 2.0) * sum(self.params.W**2.0)
        return J + Jreg

    def predict_proba(self, idx):
        """
        Predict class probabilities.
        """
        x = self.sparams.L[idx]
        p = softmax(self.params.W.dot(x) + self.params.b)
        return p

    def predict(self, idx):
        """Predict most likely class."""
        P = self.predict_proba(idx)
        return argmax(P, axis=1)