import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N   = x.shape[0]
    D   = np.prod(x.shape[1:])
    M   = b.shape[1]
    out = np.dot(x.reshape(N, D), w.reshape(D, M)) + b.reshape(1, M)
    return out, (x,w,b)

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N   = x.shape[0]
    D   = np.prod(x.shape[1:])
    M   = b.shape[1]

    dx = np.dot(dout, w.reshape(D, M).T).reshape(x.shape)
    dw = np.dot(x.reshape(N, D).T, dout).reshape(w.shape)
    db = np.sum(dout, axis=0)

    return dx, dw, db

def sigmoid_forward(x):
    """
    Computes the forward pass for a sigmoid activation.

    Inputs:
    - x: Input data, numpy array of arbitary shape;

    Returns a tuple (out, cache)
    - out: output of the same shape as x
    - cache: identical to out; required for backpropagation
    """
    return sigmoid(x), sigmoid(x)

def sigmoid_backward(dout, cache):
    """
    Computes the backward pass for an sigmoid layer.

    Inputs:
    - dout: Upstream derivative, same shape as the input
            to the sigmoid layer (x)
    - cache: sigmoid(x)
    Returns a tuple of:
    - dx: back propagated gradient with respect to x
    """
    x = cache
    return sigmoid_grad(x) * dout

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    N = data.shape[0]

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    hidden   = np.dot(data,W1) + b1
    layer1_a = sigmoid(hidden)
    layer2   = np.dot(layer1_a, W2) + b2
    # need to calculate the softmax loss
    probs = softmax(layer2)
    cost  = - np.sum(np.log(probs[np.arange(N), np.argmax(labels, axis=1)]))
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    #There is no regularization :/
    # dx -> sigmoid -> W2 * layer1_a + b -> sigmoid -> W1 * data + b1 -> ..
    dx     = probs.copy()
    dx    -= labels

    dlayer2   = np.zeros_like(dx)
    gradW2    = np.zeros_like(W2)
    gradW1    = np.zeros_like(W1)
    gradb2    = np.zeros_like(b2)
    gradb1    = np.zeros_like(b1)

    gradW2    = np.dot(layer1_a.T, dx)
    gradb2    = np.sum(dx, axis=0)
    dlayer2   = np.dot(dx, W2.T)
    dlayer1   = sigmoid_grad(layer1_a) * dlayer2
    gradW1    = np.dot(data.T, dlayer1)
    gradb1    = np.sum(dlayer1, axis=0)

    # Decided to implement affine (forward and backward function)
    #                      sigmoid (forward and backward function)
    # These should work properly;
    # scores, cache_1  = affine_forward(data, W1, b1)
    # scores, cache_s1 = sigmoid_forward(scores)
    # scores, cache_2  = affine_forward(scores, W2, b2)

    # # need to calculate the softmax loss
    # probs = softmax(scores)
    # cost  = -np.sum(np.log(probs[np.arange(N), np.argmax(labels)] + 1e-12)) / N
    # softmax_dx    = probs.copy()
    # softmax_dx[np.arange(N), np.argmax(labels,axis=1)] -= 1
    # softmax_dx /= N

    # grads = {}

    # dlayer2, grads['W2'], grads['b2'] = affine_backward(softmax_dx, cache_2)
    # dlayer1s                          = sigmoid_backward(dlayer2, cache_s1)
    # dlayer1, grads['W1'], grads['b1'] = affine_backward(dlayer1s, cache_1)
    #softmax_dx is the gradient of the loss w.r.t. y_{est}
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 300
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    #cost, _ = forward_backward_prop(data, labels, params, dimensions)
    # # expect to get 1 in 10 correct
    #print(np.exp(-cost))
    # #cost is roughly correct

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
