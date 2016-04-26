##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    eps = sqrt(6/(m+n))
    A0 = random.uniform(low=-eps, high=eps, size=(m,n))
    #### END YOUR CODE ####
    assert A0.shape == (m,n)
    return A0