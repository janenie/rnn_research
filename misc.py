##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    range_ = sqrt(6.0 / (m + n))
    A0 = random.uniform(low = -range_, high = range_, size=(m, n))
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0