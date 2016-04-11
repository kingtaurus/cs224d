'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_neural.py -vv -s -q
python -m py.test tests/test_neural.py -vv -s -q --cov

py.test.exe --cov=cs224d/ tests/test_neural.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''

import pytest
import numpy as np

import random

from collections import defaultdict, OrderedDict, Counter

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


