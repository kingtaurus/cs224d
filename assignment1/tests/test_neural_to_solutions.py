'''
HOW TO RUN THIS CODE (if tests are within the assignment 1 root):
python -m py.test tests/test_neural_to_solutions.py -vv -s -q
python -m py.test tests/test_neural_to_solutions.py -vv -s -q --cov

py.test.exe --cov=cs224d/ tests/test_neural_to_solutions.py --cov-report html

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
from q2_neural import forward_backward_prop
from q2_neural import affine_forward, affine_backward, sigmoid_forward, sigmoid_backward

from q2_neural_sol import forward_backward_prop_sol

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

@pytest.fixture(scope='module')
def construct_toy_model(D1=10, H=20, D2=10, N=100):
    dim     = [D1, H, D2]
    data    = np.random.randn(N, dim[0])
    labels  = np.zeros((N,dim[2]))
    for i in range(N):
        labels[i, np.random.randint(0, dim[2]-1)] = 0

    params = np.random.randn((dim[0] + 1) * dim[1] + (dim[1] + 1) * dim[2], )
    return data,labels,params,dim

def test_affine_forward():
    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
    w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim).reshape((1,output_dim))

    out, _ = affine_forward(x, w, b)
    correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                            [ 3.25553199,  3.5141327,   3.77273342]])

    # Compare your output with ours. The error should be around 1e-9.
    assert out.shape == correct_out.shape
    assert rel_error(out, correct_out) < 5e-7

def test_affine_backward():
    x = np.random.randn(10, 2, 3)
    w = np.random.randn(6, 5)
    b = np.random.randn(5).reshape((1,5))
    dout = np.random.randn(10, 5)

    #use eval_numerical_gradient_array for backprop from an output layer:
    # input -> layer -> output -> ... -> final_layer_loss
    # backprop becomes:
    # final_layer_loss -> gradient_of_loss (g.o.l)
    # g.o.l -> .. -> g.o.l backproped -> output -> layer -> g.o.l @ input
    dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

    _, cache = affine_forward(x, w, b)
    dx, dw, db = affine_backward(dout, cache)

    assert dx.shape == dx.shape
    assert dw.shape == dw.shape
    assert db.shape == db.shape

    assert rel_error(dx_num,dx) < 5e-7
    assert rel_error(dw_num,dw) < 5e-7
    assert rel_error(db_num,db) < 5e-7

@pytest.mark.parametrize("dim1", list(range(2,10)))
@pytest.mark.parametrize("dim2", list(range(2,10)))
@pytest.mark.parametrize("dim3", list(range(2,10)))
def test_neural_vs_neural_sol(dim1, dim2, dim3, N=300):
    dimensions = [ dim1, dim2, dim3 ]
    data = np.random.randn(N, dim1)
    labels = np.zeros((N, dim3))
    for i in range(N):
        labels[i, random.randint(0,dim3 -1)] = 1.

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    cost, grad = forward_backward_prop(data, labels, params, dimensions)
    cost_sol, grad_sol = forward_backward_prop_sol(data, labels, params, dimensions)
    assert rel_error(cost, cost_sol) < 1e-7

@pytest.mark.parametrize("dim1", list(range(2,10)))
@pytest.mark.parametrize("dim2", list(range(2,10)))
@pytest.mark.parametrize("dim3", list(range(2,10)))
def test_neural_vs_neural_sol_gradient(dim1, dim2, dim3, N=300):
    dimensions = [ dim1, dim2, dim3 ]
    data = np.random.randn(N, dim1)
    labels = np.zeros((N, dim3))
    for i in range(N):
        labels[i, random.randint(0,dim3 -1)] = 1.

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    cost, grad = forward_backward_prop(data, labels, params, dimensions)
    cost_sol, grad_sol = forward_backward_prop_sol(data, labels, params, dimensions)
    assert rel_error(grad, grad_sol) < 1e-8

