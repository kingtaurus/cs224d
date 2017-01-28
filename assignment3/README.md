[CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
=====================================================================================

Assignment #3: Recursive Neural Networks
----------------------------------------

**Due Date: 5/21/2016 (Sat) 11:59 PM PST.**

In this assignment you will learn how to use implement a Recursive Neural Net in TensorFlow.

As with previous assignments, you're limited to a maximum of three late days on this assigment.

Setup
-----

**Note**: Please be sure you have Python 2.7.x installed on your system. The following instructions should work on Mac or Linux. If you have any trouble getting set up, please come to office hours and the TAs will be happy to help.

**Get the code**: [Download the starter code here](http://cs224d.stanford.edu/assignment3/assignment3.zip) and [the assignment handout here](http://cs224d.stanford.edu/assignment3/assignment3.pdf).

**Python package requirements**: The core requirements for this assignment are
* tensorflow
* numpy

If you have a recent linux (Ubuntu 14.04 and later) install or Mac OS X, the default [`TensorFlow` installation directions](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html) will work well for you. If not, we recommend using the installation on the [corn clusters](https://web.stanford.edu/group/farmshare/cgi-bin/wiki/index.php/Main_Page). Note that you will need to use the system default python, not a local Anaconda python.

The corn clusters don't provide GPU support. If you'd like to use GPUs, we recommend using AWS. We've put together a [**brief tutorial**](http://cs224d.stanford.edu/supplementary/aws-tutorial-2.pdf) with directions on how to get started with TensorFlow on AWS.

Submitting your work
--------------------

Do not code outside of the "`# YOUR CODE HERE`", modify the list of `imports`, change function names, etc. Tuning parameters is encouraged. Make sure your code runs before submitting. Crashing due to undefined variables, missing imports, hard-coded dimensions, and bad indentation will lead to significant (non-regradable) deductions.

Once you are done working, run `./prepare_submission`. Ensure the resulting zip file is named `<your-sunet-id>.zip`, for instance if your stanford email is `jdoe@stanford.edu`, your file name should be `jdoe.zip`.

For the written component, please upload a PDF file of your solutions to Gradescope. If you are enrolled in the class you should have been signed up automatically. If you added the class late or are not signed up, post privately to Piazza and we will add you to the roster. When asked to map question parts to your PDF, please map the parts accordingly as courtesy to your TAs. This is crucial so that we can provide accurate feedback. If a question has no written component (completely programatic), map it on the same page as the previous section or next section.

Assignment Overview (Tasks)
---------------------------

This assignment is much shorter and has only one part. We recommend reading the assignment carefully and starting early as some parts may take significant time to run.

###Q1: Recursive Neural Network (100 points, 10 points extra credit)

# Extra

## Efficient recursive (tree-structured) neural networks in TensorFlow
The code for `codebase_release/rnn_tensorarray.py` is based upon the implementation from @bogatyy (see [bogatyy/cs224d](https://github.com/bogatyy/cs224d/tree/master/assignment3)).

A recursive neural network model relies upon the parsed **tree** structure of sentences and can provide strong results on sentiment analysis tasks. Since the network architecture is different for every example, it can't easily be implemented in the static graph model. The rough structure of a tree looks like the following ![recursive network](recursive.png)

## Dynamic Model
The naive way to implement a recursive tree network model is to build computational graphs per example for every tree. The construct of the graph relies upon iterating over each python tree in a depth-first search (embedding the words/leaf nodes) and the composing the resultant vectors. This can be viewed as the following recursive descent python-code:

```python
def walk(in_node):
  if in_node.isLeaf:
    return embed(in_node.word)
  left  = walk(in_node.left)
  right = walk(in_node.right)
  return compose(left, right)
```

Building the computational graph on a per tree basis, adds a bunch of new intermediate nodes to the current active session. In order to deal with this, `rnn.py` clears the default graph after some number of steps.

```python
step = 0
for step < len(train_data):
    with tf.Graph.as_default(), tf.Session() as sess:
        self.add_model_vars()
        saver = tf.train.Saver()
        #reload the model
        saver.restore(sess, weights_path)
        for r_step in range(RESET_AFTER):
            if step >= len(train_data):
                break
            train_op = get_train_op(tree[step])
            sess.run([train_op])
            step += 1
        saver.save(sess, './weights/weights_file', write_meta_graph=False)
```

## Static Computation Graph using `tf.while_loop`
Recent versions of TensorFlow provide the ability to construct dynamic graphs using `tf.TensorArray`, `tf.while_loop` and `tf.cond` (tensorflow r0.8). The initial implementation is identical as before, define and declare variables.

Currently within `rnn_while_loop_storage.py` this handled using `tf.variable_scope` and `tf.get_variable` in order to ensure encapsulation of different layer behaviours (as well as allowing a modular swapping of embedding vectors).

```python
class RNN_Model():
    def __init__(self, config):
        #.... snip ...
    #private function used to construct the graph
    def _embed_word(self, word_index):
        with tf.variable_scope("Composition", reuse=True) as scope:
             embedding = tf.get_variable("embedding")
        return tf.expand_Dims(tf.gather(embedding, word_index, 0))
    # ... snip ..
    def add_model_vars(self):
        with tf.variable_scope('Composition') as scope:
            with tf.device('/cpu:0'):
                embedding = tf.get_variable("embedding", [self.vocab.total_words,  self.config.embed_size])
            W1 = tf.get_variable("W1", [2 * self.config.embed_size, self.config.embed_size])
            b1 = tf.get_variable("b1", [1, self.config.embed_size])
            # ... l2_loss; variable summaries, etc.
```

A few details about the implementation.

(1) `W1` has size [2 * embedding_size, embedding size]: - **REASON** - this respresents `[W(left), W(right)]`, in a compact stored format.

(2) `embedding` is currently trainable (it will be learned) - **REASON** - this can be changed to a tf.get_variable(...., trainable=False) if there is already a set of word vectors available for the corpus that is being used (glove, word2vec); This can be assigned using `sess.run(embedding.assign(embedding_matrix))`;

(3) `embedding` is done on the cpu (`tf.gather`, `tf.nn.embedding_lookup`) - **REASON** - embedding operations need to be done on the cpu (there is a strong chance that these operations will not work on a gpu currently);


## Loop Operation
