import numpy as np
import tensorflow as tf

import pytest

def rel_error(x,y):
	""" returns relative error """
	return np.max(np.abs(x-y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))

@pytest.fixture(scope='module')
def array_1():
	return np.array([1,2])

