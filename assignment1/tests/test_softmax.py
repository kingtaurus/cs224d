import pytest
import numpy as np
from q1_softmax import softmax

import random

from collections import defaultdict, OrderedDict, Counter

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_assert():
    assert 1

@pytest.fixture(scope='module')
def array_1():
	return np.array([1,2])

@pytest.fixture(scope='module')
def array_2():
	return np.array([1001,1002])

def fake_data_normal(in_dim_1, in_dim_2, mean=0., sigma=1.):
	return np.random.normal(loc=mean, scale=sigma, size=(in_dim_1,in_dim_2))

def fake_data_uniform(in_dim_1, in_dim_2, low=-1000., high=1000.):
	return np.random.uniform(low=low, high=high, size=(in_dim_1, in_dim_2))

def fake_linear_shift(low=-100, high=100.):
	return np.random.uniform(low,high)

def fake_shift_vector_shift(in_dim, low=-100., high=100.):
	return np.random.uniform(low=low,high=high,size=(in_dim,1))


#starting with some simple fixed test
def test_array_1(array_1):
	assert rel_error(softmax(array_1), np.array([0.26894142,  0.73105858])) < 1e-8

def test_array_2(array_1, array_2):
	assert rel_error(softmax(array_2), softmax(array_1)) < 1e-8

@pytest.mark.parametrize("dim_1", list(range(1,20,3)))
@pytest.mark.parametrize("dim_2", list(range(1,20,3)))
def test_softmax_linearity(dim_1, dim_2):
	shift = np.random.uniform(-100,100)
	a1    = np.random.normal(size=(dim_1,dim_2))
	a2    = a1 + shift
	assert rel_error(np.max(shift), np.min(shift)) <1e-5
	assert rel_error(softmax(a1),softmax(a2)) < 1e-8

#probably can move this to a 'fake' data call
@pytest.mark.parametrize("dim_1", list(range(1,20,3)))
@pytest.mark.parametrize("dim_2", list(range(1,20,3)))
def test_softmax_linearity_rowwise(dim_1, dim_2):
	shift = np.random.uniform(low=-100,high=100,size=(dim_1,1))
	#print(shift)
	a1    = np.random.normal(size=(dim_1,dim_2))
	a2    = a1 + shift
	assert rel_error(np.max(a2 - a1), np.max(shift)) < 1e-5
	assert rel_error(softmax(a1),softmax(a2)) < 1e-8

def test_matrix_1():
    test1 = softmax(np.array([1,2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print(test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

@pytest.fixture(scope='module')
def random_2d_array():
	assert 1