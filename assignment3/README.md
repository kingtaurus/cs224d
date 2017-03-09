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
Recent versions of TensorFlow provide the ability to construct dynamic graphs using `tf.TensorArray`, `tf.while_loop` and `tf.cond` (tensorflow r0.8). The declaration of the `tf.while_loop`:

```python
def tf.while_loop(cond, body, loop_vars, 
                  shape_invariants=None,
                  parallel_iterations=10
                  back_prop=True,
                  swap_memory=False,
                  name=None)
```

Behaviour:

 (1) Repeat `body` while the condition `cond` is true;
 
 (2) `cond` is a **callable** returning a boolean scalar tensor.
 
 (3) `body` is a **callable** returning a (possibly nested) tuple, namedtupe or list of tensors of the same **arity** (**length** and **structure**) and types as `loop_vars`.
 
 (4) `loop_vars` is (possibly nested) tuple, namedtuple or list of tensors that is passed to both
    `cond` and `body`.
    
 (5) `cond` and `body` both take as many arguments as there are `loop_vars`.


In order to ensure that things make sense, `tf.while_loop()` strictly enforces shape invariants for the loop variables. A shape invariant is a (including partial) shape that is unchanged across iterations of the loop. For example, a function `f` that defined as follows:

```python
def f(x, y):
    #
    return x + 1, y - 1
```

passes the requirement for `while_loop` shape_invariants, while

```python
def f(x, y)
    return x+1
```

doesn't. The `shape_invariant` argument allows the caller to specify a less specific shape invariant for each loop variable (`[None, 3]`, if the first index could change during the loop). If not specified, the shape_invariants is equivalent to the absolute shape as specified by the `loop_vars`. So for example, `shape_invariants = [i0.get_shape(), tf.TensorShape([None, 3])]]` would be used. Some examples:

```python
i = tf.constant(0)
c = lambda i: tf.less(i,10)
b = lambda i: tf.add(i,1)
r = tf.while_loop(cond=c, body=b, loop_vars=[i])
init_op = tf.global_variable_initializer()
sess.run(init_op)
sess.run(r)
```
Returns:
```
10
```
From this, it is possible to store the data dynamically within the `tf.while_loop` iteration variable or within a `tf.TensorArray`.

The initial implementation is identical as before, define and declare variables. Currently within `rnn_while_loop_storage.py` this handled using `tf.variable_scope` and `tf.get_variable` in order to ensure encapsulation of different layer behaviours (as well as allowing a modular swapping of embedding vectors).

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

(3) ~~`embedding` is done on the cpu (`tf.gather`, `tf.nn.embedding_lookup`) - **REASON** - embedding operations need to be done on the cpu (there is a strong chance that these operations will not work on a gpu);~~ `embedding` can now be down on the GPU (this might require some modification of the code);.

## Tree Construction behaviour
Consider the tree, `(the (very (old cat)))` with four leaves, the structure of the tree looks like:

```
     *
   /   \
the
         *
       /   \
    very    *
          /   \
        old   cat
```

The vocab dictionary, would contain the word_index (the associated row of the embedding matrix that corresponds to the word vector); A represenation of this could be:

```python
vocab = {'the': 0, `very`: 1, `old`: 2, `cat`: 3}
```

The [Stanford Tree Bank](http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip) has the following structure (taken from the first line of the unzipped file `trees/train.txt`):

```
(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) (2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 's)) (2 (3 new) (2 (2 ``) (2 Conan)))))))) (2 '')) (2 and)) (3 (2 that) (3 (2 he) (3 (2 's) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash)) (2 (2 even) (3 greater)))) (2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) (2 Schwarzenegger)) (2 ,)) (2 (2 Jean-Claud) (2 (2 Van) (2 Damme)))) (2 or)) (2 (2 Steven) (2 Segal))))))))))))) (2 .)))
```

A few points about this:

(1) Sentiment of the word is the number next to the `leaf` or `node`;

(2) The structure of the `tree` is built in the same way as the simple example above;

(3) In order to convert the recursive structure to an interatively consumed structure, the leaves of any intermediate node have to be computed prior to evaulating a node. This imposes a requirement of a depth first implementation (or a stack);

Within `tree.py` there is the function:
```python
def leftTraverse(node, nodeFn=None, args=None):
    if node is None:
        return
    leftTraverse(node.left, nodeFn, args)
    leftTraverse(node.right, nodeFn, args)
    nodeFn(node, args)
```

using this, as follows:
```python
in_node = tree.root
nodes_list = list()
tr.leftTraverse(in_node, 
                lambda node, args: args.append(node),
                nodes_list
)
```

Allows us to generate a `depth and leaf first` list representation of the tree. From this representation it is possible to construct an index representation of the associated child nodes. This could be handled by in_node.index(node) or by an OrderedDict():

```python
node_to_index = OrderedDict()
for idx, i in enumerate(nodes_list):
    node_to_index[i] = idx
```

The feed dictionary (assigning values to placeholders required for RNN computation) would be constructed (for the OrderedDict solution)
```python
feed_dict = {
    self.is_a_leaf   : [ n.isLeaf for n in nodes_list ],
    self.left_child  : [ node_to_index[n.left] if not n.isLeaf else -1 for n in nodes_list ],
    self.right_child  : [ node_to_index[n.right] if not n.isLeaf else -1 for n in nodes_list ],
    self.word_index  : [ self.vocab.encode(n.word) if n.word else -1 for n in nodes_list ],
    self.labelholder : [ n.label for n in nodes_list ]
}
```
 **or**, using list indexing:

```python
feed_dict = {
    self.is_a_leaf   : [ n.isLeaf for n in nodes_list ],
    self.left_child : [ nodes_list.index(n.left) if not n.isLeaf else -1 for n in nodes_list ],
    self.right_child : [ nodes_list.index(n.right) if not n.isLeaf else -1 for n in nodes_list ],
    self.word_index  : [ self.vocab.encode(n.word) if n.word else -1 for n in nodes_list ],
    self.labelholder : [ n.label for n in nodes_list ]
}
```

## Loop Implementation
In order to construct the RNN, the following methods are required, `_embed_word`, `_combine_children`, `_loop_over_tree`, `construct_tensor_array`. The functions are implemented as follows (with the declaration of the necessary placeholders):

```python
    word_index = tf.placeholder([None], dtype=int32)
    is_a_leaf = tf.placeholder(tf.bool, [None], name="is_a_leaf")
    left_child  = tf.placeholder(tf.int32, [None], name="lchild")
    right_child = tf.placeholder(tf.int32, [None], name="rchild")
    labelholder = tf.placeholder(tf.int32, [None], name="labels_holder")

    def _embed_word(word_idx):
        with tf.variable_scope("Composition", reuse=True) as scope:
            embedding = tf.get_variable("embedding")
        return tf.expand_dims(tf.gather(embedding, word_idx), 0)

    def _combine_children(tensor_concat, left_idx, right_idx):
        left_tensor = tf.expand_dims(tf.gather(tensor_concat, left_idx), 0)
        right_tensor = tf.expand_dims(tf.gather(tensor_concat, right_idx), 0)
        with tf.variable_scope('Composition', reuse=True):
            W1 = tf.get_variable('W1')
            b1 = tf.get_variable('b1')
        return tf.nn.relu(tf.matmul(tf.concat(1, [left_tensor, right_tensor]), W1) + b1)

    def _loop_over_tree(i, tensor_list):
        is_leaf = tf.gather(is_a_leaf, i)
        word_idx    = tf.gather(word_index, i)
        left_child  = tf.gather(left_child, i)
        right_child = tf.gather(right_child, i)
        node_tensor = tf.cond(is_leaf, lambda : _embed_word(word_idx),
                                       lambda : _combine_children(tensor_list, left_child, right_child))
        tensor_list = tf.concat(0, [tensor_list, node_tensor])
        i = tf.add(i,1)
        return i, tensor_list

    def construct_tensor_array():
        loop_condition = lambda i, tensor_array: \
                         tf.less(i, tf.squeeze(tf.shape(is_a_leaf)))
        left_most_element = _embed_word(tf.gather(word_index), 0)
        i1 = tf.constant(1, dtype=tf.int32)
        while_loop_op = tf.while_loop(cond=loop_condition,
                                       body=_loop_over_tree,
                                       loop_vars=[i1, left_most_element],
                                       shape_invariants=[i1.get_shape(), tf.TensorShape([None, config.embed_size])])
        return while_loop_op[1]
```

## Results


## Pytorch Implementation
A simple `pytorch` implementation, requires a few details:

(1) Declaration of model;

(2) Declaration of recursive descent over the tree;

(3) Loss Defintion;

(4) Optimizer Declaration; 

### Pytorch model
```python
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm

EMBED_SIZE = 100
LABEL_SIZE = 2
LR = 0.01
L2 = 0.02
TRAIN_SIZE = 800

class RNN_Model(nn.Module):
  def __init__(self, vocab, embed_size=100, label_size=2):
    super(RNN_Model, self).__init__()
    self.embed_size = embed_size
    self.label_size = label_size
    self.vocab = vocab
    self.embedding = nn.Embedding(int(self.vocab.total_words), self.embed_size)
    self.fcl = nn.Linear(self.embed_size, self.embed_size, bias=True)
    self.fcr = nn.Linear(self.embed_size, self.embed_size, bias=True)
    self.projection = nn.Linear(self.embed_size, self.label_size , bias=True)
    self.activation = F.relu
    self.node_list = []
  
  def walk_tree(self, in_node):
    #defined below;
    #....

  def forward(self, x):
    """
    Forward function accepts input data and returns a Variable of output data
    """
    self.node_list = []
    root_node = self.walk_tree(x.root)
    all_nodes = torch.cat(self.node_list)
    #now I need to project out
    return all_nodes
```

### Pytorch recursive descent function

```python
  def walk_tree(self, in_node):
    if in_node.isLeaf:
      word_id = torch.LongTensor((self.vocab.encode(in_node.word), ))
      current_node = self.embedding(Variable(word_id))
      self.node_list.append(self.projection(current_node).unsqueeze(0))
    else:
      left  = self.walk_tree(in_node.left)
      right = self.walk_tree(in_node.right)
      current_node = self.activation(self.fcl(left) + self.fcl(right))
      self.node_list.append(self.projection(current_node).unsqueeze(0))
    return current_node
```

### Pytorch Loss Function

```python
#tree is a single example
#model(tree)
all_nodes = model(tree)
labels  = []
indices = []
for x,y in enumerate(tree.labels):
  if y != 2:
    labels.append(y)
    indices.append(x)
torch_labels = torch.LongTensor([l for l in labels if l != 2])
logits = all_nodes.index_select(dim=0, index=Variable(torch.LongTensor(indices)))
logits_squeezed = logits.squeeze()
predictions = logits.max(dim=2)[1].squeeze()
objective_loss = F.cross_entropy(input=logits_squeezed, target=Variable(torch_labels))
```

### Pytorch Optimizer
```python
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, dampening=0.0)
for epoch in range(max_epochs):
  for step, tree in enumerate(train_data):
     all_nodes = model(tree)
     # ... snip ... loss function above
     optimizer.zero_grad()
     objective_loss.backward()
     clip_grad_norm(model.parameters(), 5, norm_type=2.)
     optimizer.step()
```
