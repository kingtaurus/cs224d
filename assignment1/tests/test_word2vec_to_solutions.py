'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_word2vec_to_solutions.py -vv -s -q
python -m py.test tests/test_word2vec_to_solutions.py -vv -s -q --cov

py.test.exe --cov=cs224d/ tests/test_word2vec_to_solutions.py --cov-report html

(if the tests are within the subfolder tests)
PYTHONPATH=${PWD} py.test.exe tests/ -v --cov-report html
python -m pytest tests -v --cov-report html

Open index.html contained within htmlcov
'''

import pytest
import numpy as np

import random
from collections import defaultdict, OrderedDict, Counter
from q2_gradcheck import grad_numerical, eval_numerical_gradient_array

from q3_word2vec import normalizeRows
from q3_word2vec import softmaxCostAndGradient, negSamplingCostAndGradient
from q3_word2vec import skipgram, cbow

from q3_word2vec_sol import normalizeRows_sol
from q3_word2vec_sol import softmaxCostAndGradient_sol, negSamplingCostAndGradient_sol
from q3_word2vec_sol import skipgram_sol, cbow_sol

from q3_word2vec import word2vec_sgd_wrapper

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


@pytest.fixture(scope='module')
def dataset_default():
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx   = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext
    return dataset

@pytest.fixture(scope='module')
def dataset_large(size = 10):
    dataset = type('dummy', (), {})()
    assert size < 26
    def dummySampleTokenIdx():
        return random.randint(0, size+1)

    def getRandomContext(C):
        tokens = [chr(i + ord('a')) for i in range(0, size+1)]
        return tokens[random.randint(0,size+1)], [tokens[random.randint(0,size+1)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx   = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext
    return dataset


