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

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_sigmoid():
    """ Original sigmoid test defined in q2_sigmoid.py; """
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    assert rel_error(f, np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-8

def test_sigmoidgrad():
    """ Original sigmoid gradient test defined in q2_sigmoid.py; """
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    assert rel_error(g, np.array([[0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-8

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

def test_sigmoid_minus_z(count = 100):
    z = np.random.normal(loc=0., scale=100., size=count)
    y = -z
    assert rel_error(1 - sigmoid(y), z) <= 1e-8


