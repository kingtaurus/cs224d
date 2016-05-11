import numpy as np
import random

from q1_softmax_sol import softmax_sol as softmax
from q2_sigmoid_sol import sigmoid_sol as sigmoid
from q2_sigmoid_sol import sigmoid_grad_sol as sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop_sol(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    hidden = sigmoid(data.dot(W1) + b1)
    prediction = softmax(hidden.dot(W2) + b2)
    cost = -np.sum(np.log(prediction) * labels)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    delta = prediction - labels
    gradW2 = hidden.T.dot(delta)
    gradb2 = np.sum(delta, axis = 0)
    delta = delta.dot(W2.T) * sigmoid_grad(hidden)
    gradW1 = data.T.dot(delta)
    gradb1 = np.sum(delta, axis = 0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad
