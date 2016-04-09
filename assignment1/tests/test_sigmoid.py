'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_sigmoid.py -vv -s -q
python -m py.test tests/test_sigmoid.py -vv -s -q --cov

py.test.exe --cov=cs224d/ tests/test_sigmoid.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''

import pytest
import numpy as np
from q2_sigmoid import sigmoid, sigmoid_grad

import random

from collections import defaultdict, OrderedDict, Counter

COUNT=5

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-7, np.abs(x) + np.abs(y))))

def test_sigmoid():
    """ Original sigmoid test defined in q2_sigmoid.py; """
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    assert rel_error(f, np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-7

def test_sigmoidgrad():
    """ Original sigmoid gradient test defined in q2_sigmoid.py; """
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    assert rel_error(g, np.array([[0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-7

@pytest.mark.parametrize("dim", list(range(1,8)))
def test_sigmoid_shape(dim):
    testing_shape = []
    for y in range(0,dim):
        testing_shape.append(np.random.randint(3,8))
    shape = tuple(testing_shape)
    #z = np.random.randn(*testing_shape)
    x = np.random.standard_normal(shape)
    y = np.copy(x)
    assert x.shape == sigmoid(y).shape
    assert x.shape == sigmoid_grad(sigmoid(y)).shape

def test_sigmoid_minus_z(count=100):
    z = np.random.normal(loc=0., scale=100., size=count)
    y = -z
    assert rel_error(1 - sigmoid(y), sigmoid(z)) <= 1e-7

def test_sigmoid_monotone(count=100):
    z     = np.random.normal(loc=0., scale=100., size=count)
    shift = np.random.uniform(low=0., high=10., size=count)
    assert np.all(sigmoid(z + shift) - sigmoid(z)) >= 0
    assert np.all(sigmoid(z - shift) - sigmoid(z)) <= 0

def test_sigmoid_range(count=100):
    z = np.random.normal(loc=0., scale=100., size=count)
    assert np.max(sigmoid(z)) <= 1.
    assert np.max(sigmoid(z)) >= 0.

@pytest.mark.parametrize('execution_number', list(range(COUNT)))
@pytest.mark.parametrize("dim_1", list(range(1,20)))
def test_sigmoid_permutation_axis0(dim_1, execution_number):
    """ sigmoid needs to be applied element-wise;"""
    a1          = np.random.normal(size=(dim_1,1))
    s1          = sigmoid(a1)

    permutation = np.random.permutation(dim_1)
    inverse_permutation = np.argsort(permutation)

    s1_perm     = sigmoid(a1[permutation])
    assert rel_error(s1_perm[inverse_permutation], s1) <= 1e-8

@pytest.mark.parametrize("dim_1", list(range(1,20)))
def test_sigmoid_permutation_axis1(dim_1):
    a1          = np.random.normal(size=(1,dim_1))
    s1          = sigmoid(a1)

    permutation = np.random.permutation(dim_1)
    inverse_permutation = np.argsort(permutation)

    s1_perm     = sigmoid(a1.ravel()[permutation])
    assert rel_error(s1_perm.ravel()[inverse_permutation], s1) <= 1e-8
#note: permutation(sigmoid(x)) = sigmoid(permutation(x))

@pytest.mark.parametrize("dim_1", list(range(1,20)))
@pytest.mark.parametrize("dim_2", list(range(1,20)))
def test_sigmoid_gradient(dim_1, dim_2):
    a1    = np.random.normal(loc=0., scale=20., size=(dim_1,dim_2))
    shift = np.random.uniform(low=1e-9, high=1e-5, size=(dim_1,dim_2))
    ap = a1 + shift
    am = a1 - shift

    dsigmoid = (sigmoid(ap) - sigmoid(am)) / (2*shift)
    assert np.abs(np.max(dsigmoid - sigmoid_grad(sigmoid(a1)))) <= 1e-7
    assert np.abs(np.min(dsigmoid - sigmoid_grad(sigmoid(a1)))) <= 1e-7
