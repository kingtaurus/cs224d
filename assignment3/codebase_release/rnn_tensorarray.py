import sys
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil
import tensorflow as tf
import tree as tr
from utils import Vocab

from collections import OrderedDict

import seaborn as sns
sns.set_style('whitegrid')

def initialize_uninitialized_vars(session):
    uninitialized = [ var for var in tf.all_variables()
                      if not session.run(tf.is_variable_initialized(var)) ]
    session.run(tf.initialize_variables(uninitialized))

def variable_summaries(variable, name):
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(variable)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(variable - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(variable))
    tf.summary.scalar('min/' + name, tf.reduce_min(variable))
    # tf.summary.histogram(name, variable)

RESET_AFTER = 50
class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    embed_size = 50
    label_size = 2
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 30
    lr = 0.01
    l2 = 0.02
    model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights'%(embed_size, l2, lr)


#initial attempt to create graph
# currently implicitly assumes tree structure (which can't be passed into tf)
# vector_stack = tf.TensorArray(tf.float32,
#                               size=0,
#                               dynamic_size=True,
#                               clear_after_read=True,
#                               infer_shape=True)
# index = tf.placeholder(shape=(), dtype=tf.int32)

# def embed_word(word_index):
#     with tf.device('/cpu:0'):
#         with tf.variable_scope("Composition", reuse=True):
#             embedding = tf.get_variable('embedding')
#     return tf.expand_dims(tf.gather(embedding, word_index), 0)

# def combine_children(left_location, right_location):
#     with tf.variable_scope('Composition', reuse=True):
#         W1 = tf.get_variable('W1')
#         b1 = tf.get_variable('b1')
#     return tf.nn.relu(tf.matmul(tf.concat(1, [vector_stack.read(left_location), vector_stack.read(right_location)]), W1) + b1)

# tf.gather(is_leaf, index)
# #get if this a leaf

# tf.gather(word, index)
# #get the word associated

# tf.gather(left_child, index)
# tf.gather(right_child, index)

## ORIGINAL IDEA:
# def walk_node(index):
#     #tf.cond(tf.gather(isLeaf, index,), ..
#     if in_node.isLeaf is True:
#         #push the value onto the stack and return index?
#         word_id = self.vocab.encode(in_node.word)
#         print("word_id = ", word_id)
#         vector_stack.write(vector_stack.size() - 1, embed_word(word_id))
#         return vector_stack.size() - 1
#         #so we return the index
#     if in_node.isLeaf is False:
#         left_node  = walk_node(in_node.left, vocab)
#         right_node = walk_node(in_node.right, vocab)
#         vector_stack.concat(combine_children(left_node, right_node))
#         return vector_stack.size() - 1 
#         #merge the left - right pair, add it back to the stack
#     #this should never be hit(?)
#     return 0







class RNN_Model():
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.merged_summaries = None
        self.summary_writer = None
        self.is_a_leaf   = tf.placeholder(tf.bool, [None], name="is_a_leaf")
        self.left_child  = tf.placeholder(tf.int32, [None], name="lchild")
        self.right_child = tf.placeholder(tf.int32, [None], name="rchild")
        self.word_index  = tf.placeholder(tf.int32, [None], name="word_index")
        self.labelholder = tf.placeholder(tf.int32, [None], name="labels_holder")
        self.add_model_vars()
        self.tensor_array = tf.TensorArray(tf.float32,
                              size=0,
                              dynamic_size=True,
                              clear_after_read=False,
                              infer_shape=False)
        #tensor array stores the vectors (embedded or composed)
        self.tensor_array_op = None
        self.prediction   = None
        self.logits       = None
        self.root_logits  = None
        self.root_predict = None

        self.root_loss = None
        self.full_loss = None

        self.training_op = None
        #tensor_array_op is the operation on the TensorArray

    # private functions used to construct the graph.
    def _embed_word(self, word_index):
        with tf.variable_scope("Composition", reuse=True) as scope:
            print(scope.name)
            embedding = tf.get_variable("embedding")
            print(embedding.name)
        return tf.expand_dims(tf.gather(embedding, word_index), 0)

    # private functions used to construct the graph.
    def _combine_children(self, left_index, right_index):
        left_tensor  = self.tensor_array.read(left_index)
        right_tensor = self.tensor_array.read(right_index)
        with tf.variable_scope('Composition', reuse=True):
            W1 = tf.get_variable('W1')
            b1 = tf.get_variable('b1')
        return tf.nn.relu(tf.matmul(tf.concat(1, [left_tensor, right_tensor]), W1) + b1)


    # i is the index (over data stored in the placeholders)
    # identical type[out] = type[in]; can be used in while_loop
    # so first iteration -> puts left most leaf on the tensorarray (and increments i)
    # next iteration -> puts next left most (leaf on stack) and increments i
    # ....
    # until all the leaves are on the stack in the correct order
    # starts combining the leaves after and adding to the stack
    def _loop_over_tree(self, tensor_array, i):
        is_leaf     = tf.gather(self.is_a_leaf, i)
        word_idx    = tf.gather(self.word_index, i)
        left_child  = tf.gather(self.left_child, i)
        right_child = tf.gather(self.right_child, i)
        node_tensor = tf.cond(is_leaf, lambda : self._embed_word(word_idx),
                                       lambda : self._combine_children(left_child, right_child))
        tensor_array = tensor_array.write(i, node_tensor)
        i = tf.add(i,1)

        return tensor_array, i

    def construct_tensor_array(self):
        loop_condition = lambda tensor_array, i: \
                         tf.less(i, tf.squeeze(tf.shape(self.is_a_leaf)))
        #iterate over all leaves + composition
        tensor_array_op = tf.while_loop(cond=loop_condition,
                                        body=self._loop_over_tree,
                                        loop_vars=[self.tensor_array, 0],
                                        parallel_iterations=1)[0]
        return tensor_array_op

    def inference_op(self, predict_only_root=False):
        if predict_only_root:
            return self.root_logits_op()
        return self.logits_op()

    def load_data(self):
        """Loads train/dev/test data and builds vocabulary."""
        self.train_data, self.dev_data, self.test_data = tr.simplified_data(700, 100, 200)

        # build vocab from training data
        self.vocab = Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

    def add_model_vars(self):
        '''
        You model contains the following parameters:
            embedding:  tensor(vocab_size, embed_size)
            W1:         tensor(2* embed_size, embed_size)
            b1:         tensor(1, embed_size)
            U:          tensor(embed_size, output_size)
            bs:         tensor(1, output_size)
        Hint: Add the tensorflow variables to the graph here and *reuse* them while building
                the compution graphs for composition and projection for each tree
        Hint: Use a variable_scope "Composition" for the composition layer, and
              "Projection") for the linear transformations preceding the softmax.
        '''
        with tf.variable_scope('Composition') as scope:
        ### YOUR CODE HERE
        #initializer=initializer=tf.random_normal_initializer(0,3)
            print(scope.name)
            embedding = tf.get_variable("embedding",
                                        [self.vocab.total_words, self.config.embed_size])
            print(embedding.name)
            W1 = tf.get_variable("W1", [2 * self.config.embed_size, self.config.embed_size])
            b1 = tf.get_variable("b1", [1, self.config.embed_size])
            l2_loss = tf.nn.l2_loss(W1)
            tf.add_to_collection(name="l2_loss", value=l2_loss)
            variable_summaries(embedding, embedding.name)
            variable_summaries(W1, W1.name)
            variable_summaries(b1, b1.name)
        ### END YOUR CODE
        with tf.variable_scope('Projection'):
         ### YOUR CODE HERE
            U = tf.get_variable("U", [self.config.embed_size, self.config.label_size])
            bs = tf.get_variable("bs", [1, self.config.label_size])
            variable_summaries(U, U.name)
            variable_summaries(bs, bs.name)
            l2_loss = tf.nn.l2_loss(U)
            tf.add_to_collection(name="l2_loss", value=l2_loss)
        ### END YOUR CODE

    def add_model(self):
        """Recursively build the model to compute the phrase embeddings in the tree

        Hint: Refer to tree.py and vocab.py before you start. Refer to
              the model's vocab with self.vocab
        Hint: Reuse the "Composition" variable_scope here
        Hint: Store a node's vector representation in node.tensor so it can be
              used by it's parent
        Hint: If node is a leaf node, it's vector representation is just that of the
              word vector (see tf.gather()).
        Args:
            node: a Node object
        Returns:
            node_tensors: Dict: key = Node, value = tensor(1, embed_size)
        """
        if self.tensor_array_op is None:
            self.tensor_array_op = self.construct_tensor_array()
        return self.tensor_array_op

    def add_projections_op(self, node_tensors):
        """Add projections to the composition vectors to compute the raw sentiment scores

        Hint: Reuse the "Projection" variable_scope here
        Args:
            node_tensors: tensor(?, embed_size)
        Returns:
            output: tensor(?, label_size)
        """
        logits = None
        ### YOUR CODE HERE
        with tf.variable_scope("Projection", reuse=True):
            U = tf.get_variable("U")
            bs = tf.get_variable("bs")
        logits = tf.matmul(node_tensors, U) + bs
        ### END YOUR CODE
        return logits

    def logits_op(self):
        #this is an operation on the updated tensor_array
        if self.logits is None:
            self.logits = self.add_projections_op(self.tensor_array_op.concat())
        return self.logits

    def root_logits_op(self):
        #construct once
        if self.root_logits is None:
            self.root_logits = self.add_projections_op(self.tensor_array_op.read(self.tensor_array_op.size() -1))
        return self.root_logits

    def root_prediction_op(self):
        if self.root_predict is None:
            self.root_predict =  tf.squeeze(tf.argmax(self.root_logits_op(), 1))
        return self.root_predict

    def full_loss_op(self, logits, labels):
        """Adds loss ops to the computational graph.

        Hint: Use sparse_softmax_cross_entropy_with_logits
        Hint: Remember to add l2_loss (see tf.nn.l2_loss)
        Args:
            logits: tensor(num_nodes, output_size)
            labels: python list, len = num_nodes
        Returns:
            loss: tensor 0-D
        """
        if self.full_loss is None:
            loss = None
            # YOUR CODE HERE
            l2_loss = self.config.l2 * tf.add_n(tf.get_collection("l2_loss"))
            idx = tf.where(tf.less(self.labelholder,2))
            logits = tf.gather(logits, idx)
            labels = tf.gather(labels, idx)
            objective_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            loss = objective_loss + l2_loss
            tf.summary.scalar("loss_l2", l2_loss)
            tf.summary.scalar("loss_objective", tf.reduce_sum(objective_loss))
            tf.summary.scalar("loss_total", loss)
            self.full_loss = loss
        # END YOUR CODE
        return self.full_loss

    def loss_op(self, logits, labels):
        """Adds loss ops to the computational graph.

        Hint: Use sparse_softmax_cross_entropy_with_logits
        Hint: Remember to add l2_loss (see tf.nn.l2_loss)
        Args:
            logits: tensor(num_nodes, output_size)
            labels: python list, len = num_nodes
        Returns:
            loss: tensor 0-D
        """
        if self.root_loss is None:
            #construct once guard
            loss = None
            # YOUR CODE HERE
            l2_loss = self.config.l2 * tf.add_n(tf.get_collection("l2_loss"))
            objective_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            loss = objective_loss + l2_loss
            tf.summary.scalar("root_loss_l2", l2_loss)
            tf.summary.scalar("root_loss_objective", tf.reduce_sum(objective_loss))
            tf.summary.scalar("root_loss_total", loss)
            self.root_loss = loss
        # END YOUR CODE
        return self.root_loss


    def training(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.GradientDescentOptimizer for this model.
                Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: tensor 0-D
        Returns:
            train_op: tensorflow op for training.
        """
        if self.training_op is None:
        # YOUR CODE HERE
            optimizer = tf.train.AdamOptimizer(self.config.lr)#tf.train.GradientDescentOptimizer(self.config.lr)
            #optimizer = tf.train.AdamOptimizer(self.config.lr)
            self.training_op = optimizer.minimize(loss)
        # END YOUR CODE
        return self.training_op

    def predictions(self, y):
        """Returns predictions from sparse scores

        Args:
            y: tensor(?, label_size)
        Returns:
            predictions: tensor(?,1)
        """
        if self.prediction is None:
        # YOUR CODE HERE
            self.prediction = tf.argmax(y, dimension=1)
        # END YOUR CODE
        return self.prediction

    def build_feed_dict(self, in_node):
        nodes_list = []
        tr.leftTraverse(in_node, lambda node, args: args.append(node), nodes_list)
        node_to_index = OrderedDict()
        for idx, i in enumerate(nodes_list):
            node_to_index[i] = idx

        feed_dict = {
          self.is_a_leaf   : [ n.isLeaf for n in nodes_list ],
          self.left_child  : [ node_to_index[n.left] if not n.isLeaf else -1 for n in nodes_list ],
          self.right_child : [ node_to_index[n.right] if not n.isLeaf else -1 for n in nodes_list ],
          self.word_index  : [ self.vocab.encode(n.word) if n.word else -1 for n in nodes_list ],
          self.labelholder : [ n.label for n in nodes_list ]
        }
        return feed_dict

    def predict(self, trees, weights_path, get_loss = False):
        """Make predictions from the provided model."""


        results = []
        losses = []

        logits = self.root_logits_op()
        #evaluation is based upon the root node
        root_loss = self.loss_op(logits=logits, labels=self.labelholder[-1:])
        root_prediction_op = self.root_prediction_op()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, weights_path)
            for t in  trees:
                feed_dict = self.build_feed_dict(t.root)
                if get_loss:
                    root_prediction, loss = sess.run([root_prediction_op, root_loss], feed_dict=feed_dict)
                    losses.append(loss)
                    results.append(root_prediction)
                else:
                    root_prediction = sess.run(root_prediction_op, feed_dict=feed_dict)
                    results.append(root_prediction)
        return results, losses

    #need to rework this: (OP creation needs to be made independent of using OPs)
    def run_epoch(self, new_model = False, verbose=True, epoch=0):
        loss_history = []
        random.shuffle(self.train_data)
        
        with tf.Session() as sess:
            if new_model:
                add_model_op = self.add_model()
                logits = self.logits_op()
                loss = self.full_loss_op(logits=logits, labels=self.labelholder)
                train_op = self.training(loss)
                init = tf.global_variables_initializer()
                sess.run(init)
            else:
                saver = tf.train.Saver()
                saver.restore(sess, './weights/%s.temp'%self.config.model_name)
                logits = self.logits_op()
                loss = self.full_loss_op(logits=logits, labels=self.labelholder)
                train_op = self.training(loss)

            for step, tree in enumerate(self.train_data):
                feed_dict = self.build_feed_dict(tree.root)
                loss_value, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                loss_history.append(loss_value)
                if verbose:
                    sys.stdout.write('\r{} / {} :    loss = {}'.format(
                            step+1, len(self.train_data), np.mean(loss_history)))
                    sys.stdout.flush()
            saver = tf.train.Saver()
            if not os.path.exists("./weights"):
                os.makedirs("./weights")

            #print('./weights/%s.temp'%self.config.model_name)
            saver.save(sess, './weights/%s.temp'%self.config.model_name)
        train_preds, _ = self.predict(self.train_data, './weights/%s.temp'%self.config.model_name)
        val_preds, val_losses = self.predict(self.dev_data, './weights/%s.temp'%self.config.model_name, get_loss=True)
        train_labels = [t.root.label for t in self.train_data]
        val_labels = [t.root.label for t in self.dev_data]
        train_acc = np.equal(train_preds, train_labels).mean()
        val_acc = np.equal(val_preds, val_labels).mean()
        print()
        print('Training acc (only root node): {}'.format(train_acc))
        print('Valiation acc (only root node): {}'.format(val_acc))
        print(self.make_conf(train_labels, train_preds))
        print(self.make_conf(val_labels, val_preds))
        return train_acc, val_acc, loss_history, np.mean(val_losses)

    def train(self, verbose=True):
        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = -1
        for epoch in range(self.config.max_epochs):
            print('epoch %d'%epoch)
            if epoch==0:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(new_model=True, epoch=epoch)
            else:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(epoch=epoch)
            complete_loss_history.extend(loss_history)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            #lr annealing
            epoch_loss = np.mean(loss_history)
            if epoch_loss>prev_epoch_loss*self.config.anneal_threshold:
                self.config.lr/=self.config.anneal_by
                print('annealed lr to %f'%self.config.lr)
            prev_epoch_loss = epoch_loss

            #save if model has improved on val
            if val_loss < best_val_loss:
                 shutil.copyfile('./weights/%s.temp'%self.config.model_name, './weights/%s'%self.config.model_name)
                 best_val_loss = val_loss
                 best_val_epoch = epoch

            # if model has not imprvoved for a while stop
            if epoch - best_val_epoch > self.config.early_stopping:
                stopped = epoch
                #break
        if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()

        print('\n\nstopped at %d\n'%stopped)
        return {
            'loss_history': complete_loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
            }

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2, 2])
        for l,p in zip(labels, predictions):
            confmat[l, p] += 1
        return confmat


def test_RNN():
    """Test RNN model implementation.

    You can use this function to test your implementation of the Named Entity
    Recognition network. When debugging, set max_epochs in the Config object to 1
    so you can rapidly iterate.
    """
    config = Config()
    model = RNN_Model(config)
    start_time = time.time()
    stats = model.train(verbose=True)
    print('Training time: {}'.format(time.time() - start_time))

    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss_history.png")
    plt.show()

    print('Test')
    print('=-=-=')
    predictions, _ = model.predict(model.test_data, './weights/%s'%model.config.model_name)
    labels = [t.root.label for t in model.test_data]
    test_acc = np.equal(predictions, labels).mean()
    print('Test acc: {}'.format(test_acc))

if __name__ == "__main__":
        test_RNN()
