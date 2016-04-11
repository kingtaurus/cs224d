'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_gradcheck.py -vv -s -q
python -m py.test tests/test_gradcheck.py -vv -s -q --cov

py.test.exe --cov=cs224d/ tests/test_gradcheck.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''

import pytest
import numpy as np

import random

from collections import defaultdict, OrderedDict, Counter
from q2_gradcheck import grad_numerical

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

quad = lambda x: (x**2, 2*x)

def test_gradcheck_naive_1():
    """ Original sigmoid test defined in q2_sigmoid.py; """
    x = np.array(123.45)
    assert rel_error(quad(x)[1], grad_numerical(quad,x))
    
def test_gradcheck_naive_2():
    """ Original sigmoid test defined in q2_sigmoid.py; """
    x = np.random.normal(loc=10., scale=30., size=20)
    assert rel_error(quad(x)[1], grad_numerical(quad,x))

def test_gradcheck_naive_3():
    """ Original sigmoid test defined in q2_sigmoid.py; """
    x = np.random.normal(loc=10., scale=30., size=(20,20))
    assert rel_error(quad(x)[1], grad_numerical(quad,x))



