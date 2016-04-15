'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_sgd.py -vv -s -q
python -m py.test tests/test_sgd.py -vv -s -q --cov

py.test.exe --cov=cs231n/ tests/test_sgd.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''

import pytest
import numpy as np
import random

from collections import defaultdict, OrderedDict, Counter
from q3_sgd      import sgd

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

@pytest.fixture(scope='module')
def quad():
	return lambda x: (np.sum(x**2), x * 2)

def test_sgd_1(quad):
    """ Original normalization test defined in q3_word2vec.py; """

    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=None)
    assert abs(t1) <= 1e-6

def test_sgd_2(quad):
    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=None)
    assert abs(t2) <= 1e-6

def test_sgd_3(quad):
    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=None)
    assert abs(t3) <= 1e-6
