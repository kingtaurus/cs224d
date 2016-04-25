import sys
from numpy import *
import numpy as np
import itertools
import time

# math helpers
from math import *

class PackedVector(object):
    _name_to_idx = {}
    _views = []

    def __init__(self, *shapes, **shapemap):
        # Prepend named shapes
        names = range(len(shapes))
        if len(shapemap) > 0:
            nn, ss = zip(*shapemap.items())
            names = names + list(nn)
            shapes = shapes + ss
        self._name_to_idx = {n:i for i,n in enumerate(names)}

        # Generate endpoints
        self._dims = shapes
        self._lens = map(prod, self._dims)
        self._ends = concatenate([[0], cumsum(self._lens)])
        self._vec = zeros(self._ends[-1]) # allocate storage

        # Generate view objects
        self._views = []
        for i, dim in enumerate(self._dims):
            start = self._ends[i]
            end = self._ends[i+1]
            self._views.append(self._vec[start:end].reshape(dim))

        # Add view for full params
        self._name_to_idx['full'] = len(self._views)
        self._views.append(self._vec[:]) # full

        # Generate attributes for direct access
        for n,i in self._name_to_idx.iteritems():
            object.__setattr__(self, str(n), self._views[i])

    # Overload setattr to write to views
    def __setattr__(self, name, value):
        if name in self._name_to_idx:
            v = self._views[self._name_to_idx[name]]
            v[:] = value # in-place update
        else: # default behavior
            object.__setattr__(self, name, value)

    ##
    # Dictionary-style access
    def __getitem__(self, key):
        key = self._name_to_idx[key]
        return self._views[key]

    def __setitem__(self, key, value):
        key = self._name_to_idx[key]
        self._views[key][:] = value # in-place update

    def names(self):
        return [k for k in self._name_to_idx.keys() if not k == 'full']

    def reset(self):
        self.full.fill(0)

    def __repr__(self):
        listings = ["%s = \n%s" % (n, repr(self._views[i]))
                    for (n,i) in sorted(self._name_to_idx.items())
                    if n != 'full']
        return "PackedVector(\n" + "\n".join(listings) + "\n)"


class SparseDelta(object):
    """
    Wrapper class for sparse updates;
    stores a list of (idx, value) tuples,
    while supporting dict-like syntax.

    idx can be any python object, so complex
    slicing/selection like [:,range(5)] will be
    handled properly, as this is just forwarded
    to the NumPy array later.
    """

    _data = []
    def __init__(self):
        self._data = []

    def __setitem__(self, key, value):
        self._data.append((key, value))

    def __iter__(self): # iterate through managed list
        return self._data.__iter__()

    def coalesce(self):
        """
        Sum all updates with the same index.
        This is O(n^2) in the number of updates stored,
        so potentially slow - only use for grad_check.
        """
        # self._data = sorted(self._data)
        N = len(self._data)
        newdata = []
        for i in range(N):
            k1,v1 = self._data[i]
            # if k1 == None: continue
            if k1 is None: continue
            for j in range(i+1,N):
                k2,v2 = self._data[j]
                if k2 == k1:
                    v1 += v2 # combine updates
                    self._data[j] = (None, None)
            newdata.append((k1, v1))
        self._data = newdata

    def __repr__(self):
        return "SparseDelta(" + self._data.__repr__() + ")"

    def reset(self):
        self._data = []


class SparseDeltas(object):
    """
    Sparse update manager, compatible with PackedVector.

    Designed as an efficient implementation of a block-sparse
    matrix in a dictionary-of-keys style, although with a restricted
    set of features intended for use in managing gradients.

    Stores a collection of SparseDelta objects that parallels
    the views of a a PackedVector object. Each SparseDelta manages
    pairs [(idx, array), ...] for a given matrix, where idx is a rich
    indexing object that can be specified with full slicing semantics
    and used to selectively update an arbitary component of the target
    matrix.

    This should be useful for updating e.g. word vector representations
    in an efficient manner during SGD.

    Accumulate updates as:
    pv = PackedVector(**shapemap)
    sd = SparseDeltas(**shapemap)
    sd.W[idx] = value
    sd.apply_to(pv)
    sd.reset()
    """

    def __init__(self, *shapes, **shapemap):
        # Prepend named shapes
        names = range(len(shapes))
        if len(shapemap) > 0:
            nn, ss = zip(*shapemap.items())
            names = names + list(nn)
        self._names = set(map(str, names))

        # Generate attributes for direct access
        for n in self._names:
            s = SparseDelta()
            setattr(self, n, s)

    def __getitem__(self, key):
        if key in self._names:
            return getattr(self, key)
        else: raise ValueError("Key %s not found." % key)

    def coalesce(self):
        for n in self._names:
            self[n].coalesce()

    def reset(self):
        for n in self._names:
            self[n].reset()

    def names(self):
        return list(self._names)

    def apply_to(self, pv, alpha=-1.0):
        """Apply sparse updates to parameter store."""
        # assert(type(pv) == PackedVector)

        for n in self._names:
            ud = self[n] # update dict
            for idx, v in ud: # idx, vec pairs
                pv[n][idx] += alpha*v # in-place update


    def __repr__(self):
        elems = "\n".join(str(n) + " = " + repr(self[n])
                          for n in self._names)
        return "SparseDeltas(\n" + elems + "\n)"



class NNBase(object):
    """
    Versatile base class for Neural Network models.
    Implements generic parameter management, supporting
    both densely-updated parameters (e.g. weight matricies)
    and sparsely-updated parameters (e.g. representations),
    and allowing for generic learning algorithms to be
    implemented for a wide variety of NNs based on this class.

    This implements training routines and gradient checks,
    and need only be suppleme

    Subclass must implement, at minimum:
        - _acc_grads()
        - predict()
        - compute_loss()

    And should also implement an __init__() method
    that calls super.__init__() with an appropriate
    set of parameters:
        - param_dims_dense = {"W": [100x150], "b1": [100,], ...}
        - param_dims_sparse = {"L": [40000x50], ...} (optional)
        - hypers = (optional) dictionary of additional hyperparameters
    """

    def __init__(self, param_dims_dense,
                 param_dims_sparse={}, **hypers):
        # Set up hyperparameters
        for p, v in hypers:
            setattr(self, p, v)

        ##
        # Set up parameters for dense updates
        self._param_dims_dense = param_dims_dense
        self.params = PackedVector(**param_dims_dense)
        self.grads = PackedVector(**param_dims_dense)

        ##
        # Set up parameters for sparse updates
        self._param_dims_sparse = param_dims_sparse
        self.sparams = PackedVector(**param_dims_sparse)
        self.sgrads = SparseDeltas(**param_dims_sparse)



    def _reset_grad_acc(self):
        """Reset accumulated gradients."""
        self.grads.reset()
        self.sgrads.reset()

    def _acc_grads(self, x, y):
        """
        Accumulate gradients, given data -> response.

        Subclass must implement this to be train-able.
        """
        raise NotImplementedError("_acc_grads not yet implemented")

    def _apply_grad_acc(self, alpha=1.0):
        """
        Update parameters with accumulated gradients.

        alpha can be a scalar (as in SGD), or a vector
        of the same length as the full concatenated
        parameter vector (as in e.g. AdaGrad)
        """
        # Dense updates
        self.params.full -= alpha * self.grads.full
        # Sparse updates
        self.sgrads.apply_to(self.sparams, alpha=-1*alpha)


    def train_point_sgd(self, x, y, alpha):
        """Generic single-point SGD"""
        self._reset_grad_acc()
        self._acc_grads(x, y)
        self._apply_grad_acc(alpha)

    def train_minibatch_sgd(self, X, y, alpha):
        """
        Generic minibatch SGD
        """
        self._reset_grad_acc()
        for i in range(len(y)):
            self._acc_grads(X[i], y[i])
        self._apply_grad_acc(alpha)


    def grad_check(self, x, y, eps=1e-4, tol=1e-6,
                   outfd=sys.stderr, verbose=False,
                   skiplist=[]):
        """
        Generic gradient check: uses current params
        aonround a specific data point (x,y)

        This is implemented for diagnostic purposes,
        and is not optimized for speed. It is recommended
        to run this on a couple points to test a new
        neural network implementation.
        """
        # Accumulate gradients in self.grads
        self._reset_grad_acc()
        self._acc_grads(x, y)
        self.sgrads.coalesce() # combine sparse updates

        ##
        # Loop over dense parameters
        for name in self.params.names():
            if name in skiplist: continue
            theta = self.params[name]
            grad_computed = self.grads[name]
            grad_approx = zeros(theta.shape)
            for ij, v in ndenumerate(theta):
                tij = theta[ij]
                theta[ij] = tij + eps
                Jplus  = self.compute_loss(x, y)
                theta[ij] = tij - eps
                Jminus = self.compute_loss(x, y)
                theta[ij] = tij # reset
                grad_approx[ij] = (Jplus - Jminus)/(2*eps)
            # Compute Frobenius norm
            grad_delta = linalg.norm(grad_approx - grad_computed)
            print >> outfd, "grad_check: dJ/d%s error norm = %.04g" % (name, grad_delta),
            print >> outfd, ("[ok]" if grad_delta < tol else "**ERROR**")
            print >> outfd, "    %s dims: %s = %d elem" % (name, str(list(theta.shape)), prod(theta.shape))
            if verbose and (grad_delta > tol): # DEBUG
                print >> outfd, "Numerical: \n" + str(grad_approx)
                print >> outfd, "Computed:  \n" + str(grad_computed)

        ##
        # Loop over sparse parameters
        for name in self.sparams.names():
            if name in skiplist: continue
            theta_full = self.sparams[name]
            idxblocks = np.indices(theta_full.shape)
            # Loop over all sparse updates for this parameter
            for idx, grad_computed in self.sgrads[name]:
                # For arbitary indexing, might not get a contiguous block
                # therefore, can't use views for aliasing here
                # Solution: generate index arrays, select indices
                # then use these for sparse grad check
                idxtuples = zip(*[d[idx].flat for d in idxblocks])
                # idxtuples = zip(*[idxblocks[i][idx].flat
                #                   for i in range(idxblocks.shape[0])])

                # if name == "L": import pdb; pdb.set_trace() # DEBUG

                grad_approx = zeros(len(idxtuples))
                theta = theta_full # alias full
                for k, ij in enumerate(idxtuples):
                    tij = theta[ij]
                    theta[ij] = tij + eps
                    Jplus  = self.compute_loss(x, y)
                    theta[ij] = tij - eps
                    Jminus = self.compute_loss(x, y)
                    theta[ij] = tij # reset
                    grad_approx[k] = (Jplus - Jminus)/(2*eps)
                # Thankfully, numpy is *very* consistent about index order
                # and so this will put all the above indices in the right place!
                # idxtuples (i,j,k,...) are sorted by i, then j, then k, ...
                # and so will be packed properly in row-major order to match
                # the old array slice we took.
                grad_approx = grad_approx.reshape(grad_computed.shape)

                ##
                # Old version here breaks on idx = [1,2,3]
                # theta = theta_full[idx] # view of update block
                # grad_approx = zeros(theta.shape)

                # # Loop over all indices within update block
                # for ij, v in ndenumerate(theta):
                #     tij = theta[ij]
                #     theta[ij] = tij + eps
                #     Jplus  = self.compute_loss(x, y)
                #     theta[ij] = tij - eps
                #     Jminus = self.compute_loss(x, y)
                #     theta[ij] = tij # reset
                #     grad_approx[ij] = (Jplus - Jminus)/(2*eps)
                # Compute Frobenius norm
                grad_delta = linalg.norm(grad_approx - grad_computed)
                print >> outfd, "grad_check: dJ/d%s[%s] error norm = %.04g" % (name, idx, grad_delta),
                print >> outfd, ("[ok]" if grad_delta < tol else "**ERROR**")
                print >> outfd, "    %s[%s] dims: %s = %d elem" % (name, idx, str(list(grad_computed.shape)), prod(grad_computed.shape))
                if verbose and (grad_delta > tol): # DEBUG
                    print >> outfd, "Numerical: \n" + str(grad_approx)
                    print >> outfd, "Computed:  \n" + str(grad_computed)

        self._reset_grad_acc()


    def predict_proba(self, X):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.
        """
        raise NotImplementedError("predict_proba not yet implemented")

    def predict(self, X):
        """
        Predict output (score, class, etc.)
        """
        raise NotImplementedError("predict not yet implemented")

    def compute_loss(self, X, y):
        """Compute loss over data X,y"""
        raise NotImplementedError("compute_loss not yet implemented")

    def compute_mean_loss(self, X, y):
        return self.compute_loss(X, y) / len(y)

    def compute_display_loss(self, X, y):
        """
        Optional alternative loss function for printing or diagnostics.
        """
        return self.compute_mean_loss(X, y)

    def train_sgd(self, X, y,
                  idxiter=None, alphaiter=None,
                  printevery=10000, costevery=10000,
                  devidx=None):
        if idxiter == None: # default training schedule
            idxiter = xrange(len(y))
        if alphaiter == None: # default training schedule
            alphaiter = itertools.repeat(self.alpha)

        costs = []
        counter = 0
        t0 = time.time()

        try:
            print "Begin SGD..."
            for idx, alpha in itertools.izip(idxiter, alphaiter):
                if counter % printevery == 0:
                    print "  Seen %d in %.02f s" % (counter, time.time() - t0)
                if counter % costevery == 0:
                    if devidx != None:
                        cost = self.compute_display_loss(X[devidx], y[devidx])
                    else: cost = self.compute_display_loss(X, y)
                    costs.append((counter, cost))
                    print "  [%d]: mean loss %g" % (counter, cost)

                if hasattr(idx, "__iter__") and len(idx) > 1: # if iterable
                    self.train_minibatch_sgd(X[idx], y[idx], alpha)
                elif hasattr(idx, "__iter__") and len(idx) == 1: # single point
                    idx = idx[0]
                    self.train_point_sgd(X[idx], y[idx], alpha)
                else:
                    self.train_point_sgd(X[idx], y[idx], alpha)

                counter += 1
        except KeyboardInterrupt as ke:
            """
            Allow manual early termination.
            """
            print "SGD Interrupted: saw %d examples in %.02f seconds." % (counter, time.time() - t0)
            return costs

        # Wrap-up
        if devidx != None:
            cost = self.compute_display_loss(X[devidx], y[devidx])
        else: cost = self.compute_display_loss(X, y)
        costs.append((counter, cost))
        print "  [%d]: mean loss %g" % (counter, cost)
        print "SGD complete: %d examples in %.02f seconds." % (counter, time.time() - t0)

        return costs


    @staticmethod
    def epochiter(N, nepoch=5):
        """Iterator to loop sequentially through training sets."""
        return itertools.chain.from_iterable(
                    itertools.repeat(xrange(N), nepoch))

    @staticmethod
    def randomiter(N, high, batch=1):
        """Iterator to generate random minibatches."""
        for i in xrange(N):
            yield random.randint(0, high, size=batch)

    @staticmethod
    def annealiter(a0, epoch=10000):
        """
        Iterator to anneal learning rate.
        Steps down in a harmonic series after each epoch.
        So, annealiter(1.0, epoch=2) will yield
        1.0 1.0 0.5 0.5 0.33 0.33 0.25 0.25 ...
        """
        ctr = 0
        while True:
            yield a0 * 1.0/((ctr+epoch)/epoch)
            ctr += 1
