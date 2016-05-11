'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_softmax_regression.py -vv -s -q
python -m py.test tests/test_softmax_regression.py -vv -s -q --cov

py.test.exe --cov=cs224d/ tests/test_softmax_regression.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''

import pytest
import numpy as np
from q1_softmax import softmax
from q2_gradcheck import grad_numerical, eval_numerical_gradient_array
from q4_softmaxreg import softmaxRegression, accuracy, softmax_wrapper, getSentenceFeature

import random

from collections import defaultdict, OrderedDict, Counter

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

@pytest.fixture(scope='module')
def array_1():
	return np.array([1,2])

@pytest.fixture(scope='module')
def array_2():
	return np.array([1001,1002])

@pytest.fixture(scope='module')
def array_3():
	return np.array([-1001,-1002])

@pytest.fixture(scope='module')
def fake_data_normal(in_dim_1, in_dim_2, mean=0., sigma=1.):
	return np.random.normal(loc=mean, scale=sigma, size=(in_dim_1,in_dim_2))

@pytest.fixture(scope='module')
def fake_data_uniform(in_dim_1, in_dim_2, low=-1000., high=1000.):
	return np.random.uniform(low=low, high=high, size=(in_dim_1, in_dim_2))

@pytest.fixture(scope='module')
def linear_shift(low=-100, high=100.):
	return np.random.uniform(low,high)

@pytest.fixture(scope='module')
def vector_shift(in_dim, low=-100., high=100.):
	return np.random.uniform(low=low,high=high,size=(in_dim,1))

#starting with some simple fixed test
def test_softmax_array_1(array_1):
	""" Original softmax test defined in q2_softmax.py; """
	assert rel_error(softmax(array_1), np.array([0.26894142,  0.73105858])) < 1e-8

def test_softmax_array_2(array_1, array_2):
	""" Original softmax test defined in q2_softmax.py; """
	assert rel_error(softmax(array_2), softmax(array_1)) < 1e-8

def test_softmax_array_3(array_3):
	""" Original softmax test defined in q2_softmax.py; """
	assert rel_error(softmax(array_3), np.array(
        [0.73105858, 0.26894142]))

@pytest.mark.parametrize("dim_1", list(range(1,20)))
@pytest.mark.parametrize("dim_2", list(range(1,20)))
def test_softmax_shape(dim_1, dim_2):
	a1 = np.random.normal(size=(dim_1,dim_2))
	assert a1.shape == softmax(a1).shape

@pytest.mark.parametrize("dim_1", list(range(1,20,3)))
@pytest.mark.parametrize("dim_2", list(range(1,20,3)))
def test_softmax_linearity(dim_1, dim_2):
	shift = linear_shift(-100,100)
	a1    = np.random.normal(size=(dim_1,dim_2))
	a2    = a1 + shift
	assert rel_error(np.max(shift), np.min(shift)) <1e-8
	assert rel_error(softmax(a1),softmax(a2)) < 1e-8

@pytest.mark.parametrize("dim_1", list(range(1,20)))
def test_softmax_permutation_axis0(dim_1):
	a1          = np.random.normal(size=(dim_1,1))
	s1          = softmax(a1)

	permutation = np.random.permutation(dim_1)
	inverse_permutation = np.argsort(permutation)

	s1_perm     = softmax(a1[permutation])
	assert rel_error(s1_perm[inverse_permutation], s1) <= 1e-8

@pytest.mark.parametrize("dim_1", list(range(1,20)))
def test_softmax_permutation_axis1(dim_1):
	a1          = np.random.normal(size=(1,dim_1))
	s1          = softmax(a1)

	permutation = np.random.permutation(dim_1)
	inverse_permutation = np.argsort(permutation)

	s1_perm     = softmax(a1.ravel()[permutation])
	assert rel_error(s1_perm.ravel()[inverse_permutation], s1) <= 1e-8
#note: permutation(softmax(x)) = softmax(permutation(x))

#probably can move this to a 'fake' data call
@pytest.mark.parametrize("dim_1", list(range(1,20,3)))
@pytest.mark.parametrize("dim_2", list(range(1,20,3)))
def test_softmax_linearity_rowwise(dim_1, dim_2):
	shift = np.random.uniform(low=-100,high=100,size=(dim_1,1))
	#print(shift)
	a1    = np.random.normal(size=(dim_1,dim_2))
	a2    = a1 + shift
	assert rel_error(np.max(a2 - a1), np.max(shift)) < 1e-8
	assert rel_error(softmax(a1),softmax(a2)) < 1e-8

@pytest.mark.parametrize("samples", [5000, 10000])
@pytest.mark.parametrize("features", list(range(20, 41, 5)))
@pytest.mark.parametrize("classes", [100])
def test_softmax_regression(samples, features, classes):
    dummy_weights  = 0.1 * np.random.randn(features, classes) + 1.0
    dummy_features = np.random.randn(samples, features)
    dummy_labels   = np.argmax(np.random.randn(samples, classes), axis=1)
    #np.sqrt(p * (1 - p) / N )
    #n/N
    #rough estimate of how many should be correct (assuming random: -np.log(p), where p is the probability of guessing correctly)
    assert np.abs(-np.log(1./classes) - (softmaxRegression(dummy_features, dummy_labels, dummy_weights, 0.0)[0])) / -np.log(1./classes) <= 0.15 

@pytest.mark.parametrize("samples", [50, 60])
@pytest.mark.parametrize("features", list(range(20, 41, 5)))
@pytest.mark.parametrize("classes", [10])
def test_softmax_gradient(samples, features, classes, check_count=20):
    dummy_weights  = 0.1 * np.random.randn(features, classes) + 1.0
    dummy_features = np.random.randn(samples, features)
    dummy_labels   = np.argmax(np.random.randn(samples, classes), axis=1)
    #rough estimate of how many should be correct (assuming random: -np.log(p), where p is the probability of guessing correctly)
    #assert np.abs(-np.log(1./classes) - (softmaxRegression(dummy_features, dummy_labels, dummy_weights, 0.0)[0])) / -np.log(1./classes) <= 0.15 

    f = lambda w: softmaxRegression(dummy_features, dummy_labels, w, regularization=0.0, nopredictions=False)[0]
    g = lambda w: softmaxRegression(dummy_features, dummy_labels, w, regularization=0.0, nopredictions=False)[1]

    W = dummy_weights
    grad_analytic = g(W)

    for i in range(check_count):
        ix = tuple([random.randrange(m) for m in W.shape])
        shift = np.zeros(W.shape)
        shift[ix] = 1e-7
        grad_numerical = (f(W + shift) - f(W - shift)) / (2 * 1e-7)
        assert( abs(grad_numerical - grad_analytic[ix]) / (abs(grad_numerical) + abs(grad_analytic[ix])) < 0.0002)
        #only evaluating at one point (expect the majority of the deviation at this point)
