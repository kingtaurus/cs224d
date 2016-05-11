import numpy as np

def sigmoid_sol(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    x = 1. / (1 + np.exp(-x))
    ### END YOUR CODE
    return x

def sigmoid_grad_sol(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x.
    """
    ### YOUR CODE HERE
    f = f * (1-f)
    ### END YOUR CODE
    return f

