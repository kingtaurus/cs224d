'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_normalize.py -vv -s -q
python -m py.test tests/test_normalize.py -vv -s -q --cov

py.test.exe --cov=cs231n/ tests/test_normalize.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''

import pytest
import numpy as np
import random

from collections import defaultdict, OrderedDict, Counter
from q3_word2vec import normalizeRows, l1_normalize_rows, l2_normalize_rows

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_normalize():
    """ Original normalization test defined in q3_word2vec.py; """
    x      = np.array([[3.0,4.0],[1, 2]])
    norm_x = normalizeRows(x)
    y      = np.array([[0.6, 0.8], [0.4472, 0.8944]])
    assert rel_error(norm_x, y) <= 1e-4

def test_l2_normalize():
    x      = np.array([[3.0,4.0],[1, 2]])
    norm_x = l2_normalize_rows(x)
    y      = np.array([[0.6, 0.8], [0.4472, 0.8944]])
    assert rel_error(norm_x, y) <= 1e-4

@pytest.fixture(scope='module')
def test_array():
    def functor(in_dim_1 = 10, in_dim_2 = 10):
        assert in_dim_1 > 0 and in_dim_2 > 0
        return np.random.uniform(low=0.,high=10.,size=(in_dim_1,in_dim_2))
    return functor

def test_l2_against_sklearn(test_array):
    try:
        from sklearn.preprocessing import normalize
        in_array = test_array()
        assert rel_error(l2_normalize_rows(in_array), normalize(in_array, axis=1, norm='l2')) <= 1e-8
    except ImportError:
        assert 1
        print("ImportError (sklearn) on current node!")

def test_l1_against_sklearn(test_array):
    try:
        from sklearn.preprocessing import normalize
        in_array = test_array()
        assert rel_error(l1_normalize_rows(in_array), normalize(in_array, axis=1, norm='l1')) <= 1e-8
    except ImportError:
        assert 1
        print("ImportError (sklearn) on current node!")
