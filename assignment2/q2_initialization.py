import numpy as np
import tensorflow as tf

def xavier_weight_init():
  """
  Returns function that creates random tensor. 

  The specified function will take in a shape (tuple or 1-d array) and must
  return a random tensor of the specified shape and must be drawn from the
  Xavier initialization distribution.

  Hint: You might find tf.random_uniform useful.
  """
  def _xavier_initializer(shape, **kwargs):
    """Defines an initializer for the Xavier distribution.

    This function will be used as a variable scope initializer.

    https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope

    Args:
      shape: Tuple or 1-d array that species dimensions of requested tensor.
    Returns:
      out: tf.Tensor of specified shape sampled from Xavier distribution.
    """
    ### YOUR CODE HERE
    eps = 4 * np.sqrt(6 / np.sum(shape))
    out = tf.random_uniform(shape=shape, minval=-eps, maxval=eps, dtype=tf.float32)
    ### END YOUR CODE
    return out
  # Returns defined initializer function.
  return _xavier_initializer

def test_initialization_basic():
  """
  Some simple tests for the initialization.
  """
  print( "Running basic tests...")
  xavier_initializer = xavier_weight_init()
  shape = (1,)
  xavier_mat = xavier_initializer(shape)
  assert xavier_mat.get_shape() == shape

  shape = (1, 2, 3)
  xavier_mat = xavier_initializer(shape)
  assert xavier_mat.get_shape() == shape
  print( "Basic (non-exhaustive) Xavier initialization tests pass\n")

def test_initialization():
  """ 
  Use this space to test your Xavier initialization code by running:
      python q1_initialization.py 
  This function will not be called by the autograder, nor will
  your tests be graded.
  """
  print( "Running your tests...")
  ### YOUR CODE HERE
  xavier_initializer = xavier_weight_init()
  sess = tf.Session()

  shape = (100,100)
  tf_xavier = xavier_initializer(shape)
  sess.run(tf_xavier.initializer)
  xavier = sess.run(tf_xavier)
  # print(np.mean(xavier))
  # print(np.max(xavier))
  # print(np.min(xavier))
  # print(np.std(xavier))
  # eps = np.sqrt(6/np.sum((100,100)))
  # print((2 * eps)/np.sqrt(12))
  # expect min to roughly be -np.sqrt(6)/sqrt(200)
  # expect max to roughly be  np.sqrt(6)/sqrt(200)
  # expect mean to be roughly 0.
  # expect variance to be (b - a) ** 2 / 12

  ### END YOUR CODE  

if __name__ == "__main__":
    test_initialization_basic()
    test_initialization()
