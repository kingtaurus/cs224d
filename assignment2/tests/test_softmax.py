import numpy as np
import tensorflow as tf

import pytest

from q1_softmax import softmax, cross_entropy_loss

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x-y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))

@pytest.fixture(scope='module')
def array_1():
    return np.array([1,2], dtype=np.float32)

@pytest.fixture(scope='module')
def array_2():
    return np.array([1001,1002], dtype=np.float32)

@pytest.fixture(scope='module')
def array_3(array_2 = array_2()):
    return np.array([array_2], dtype=np.float32)

@pytest.fixture(scope='module')
def array_4(array_1 = array_1(), array_2 = array_2()):
    return np.array([array_1, array_2], dtype=np.float32)

@pytest.fixture(scope='module')
def CE_arrays():
    return np.array([[0, 1], [1, 0], [1, 0]]), np.array([[.5, .5], [.5, .5], [.5, .5]])

#this should construct a single tf session per function call
@pytest.fixture(scope='function')
def sess():
    return tf.Session()

def test_softmax_array_1(array_1):
    """ Original softmax test defined in q2_softmax.py; """
    with tf.Session():
        input_array = tf.convert_to_tensor(array_1)
        assert rel_error(softmax(input_array).eval(),
                         np.array([0.26894142,  0.73105858])) < 1e-7

def test_softmax_array_alt(sess, array_1):
    input_array  = tf.convert_to_tensor(array_1)
    output_array = sess.run(softmax(input_array)) 
    assert rel_error(output_array,
                     np.array([0.26894142,  0.73105858])) < 1e-7

@pytest.mark.parametrize("input_array", [array_1(), array_2(), array_3(), array_4()])
def test_get_session(sess, input_array):
    sess.run(softmax(tf.convert_to_tensor(input_array)))
    assert 1
    print("softmax ran to completion")

def test_CE_loss(sess, CE_arrays):
    y, y_hat = CE_arrays
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    y_hat = tf.convert_to_tensor(y_hat, dtype=tf.float64)
    sess.run(cross_entropy_loss(y,y_hat))
    assert 1
    print("CE_loss ran to completion")

def test_CE_loss_validation(sess, CE_arrays):
    y, y_hat = CE_arrays
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    y_hat = tf.convert_to_tensor(y_hat, dtype=tf.float64)
    value = sess.run(cross_entropy_loss(y,y_hat))
    assert rel_error(value, -3 * np.log(0.5)) <= 1e-7

