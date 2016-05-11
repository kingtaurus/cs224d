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
    dataset.genTokens        = ["a", "b", "c", "d", "e"]
    dataset.dummy_tokens     = dict((i,j) for j,i in enumerate(dataset.genTokens()))
    dataset.dummy_vectors    = normalizeRows(np.random(10,3))
    return dataset

@pytest.fixture(scope='module')
def dataset_large(size = 10):
    assert size < 26
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, size)

    def gen_tokens():
        tokens = [chr(i + ord('a')) for i in range(0, size)]
        return tokens

    def getRandomContext(C = size):
        tokens = gen_tokens()
        return tokens[random.randint(0,size-1)], [tokens[random.randint(0,size-1)] \
           for i in range(2*C)]

    dataset.size             = size
    dataset.sampleTokenIdx   = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext
    dataset.genTokens        = gen_tokens()
    dataset.dummy_tokens     = dict((i,j) for j,i in enumerate(dataset.genTokens))
    dataset.dummy_vectors    = normalizeRows(np.random.randn(size * 2, 5))
    return dataset


def test_skipgram_to_solutions(dataset_large):
    word2vec_sgd_wrapper(skipgram, dataset_large.dummy_tokens, dataset_large.dummy_vectors, dataset_large, 5)
    #this might be harder than it looks (since the number of calls to random have to be identical;)
    #will probably need to re-work this