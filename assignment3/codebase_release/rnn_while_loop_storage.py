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

from profilehooks import profile, timecall

import seaborn as sns
sns.set_style('whitegrid')

global_step = tf.Variable(0, name='global_step', trainable=False)

# def initialize_uninitialized_vars(session):
#     uninitialized = [ var for var in tf.all_variables()
#                       if not session.run(tf.is_variable_initialized(var)) ]
#     session.run(tf.initialize_variables(uninitialized))

def variable_summaries(variable, name):
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(variable)
    tf.summary.scalar(name='mean/' + name, tensor=mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(variable - mean)))
    tf.summary.scalar(name='stddev/' + name, tensor=stddev)
    tf.summary.scalar(name='max/' + name, tensor=tf.reduce_max(variable))
    tf.summary.scalar(name='min/' + name, tensor=tf.reduce_min(variable))
    #tf.summary.histogram(name, variable)

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    embed_size = 128
    label_size = 2
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 30
    lr = 0.01
    l2 = 0.02
    model_name = 'rnn_embed_dim_%d_l2_weight_%.2f_lr_%.2f.weights'%(embed_size, l2, lr)
    #probably should changes this to use directories

# saver = tf.train.import_meta_graph('results/model.ckpt-1000.meta')

# # We can now access the default graph where all our metadata has been loaded
# graph = tf.get_default_graph()
# global_step_tensor = graph.get_tensor_by_name('loss/global_step:0')
# train_op = graph.get_operation_by_name('loss/train_op')
# hyperparameters = tf.get_collection('hyperparameters')

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

        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        self.l2_reg  = tf.placeholder(tf.float32, (), name="l2_regularization_weight")

        self.add_model_vars()
        #tensor array stores the vectors (embedded or composed)
        self.tensor_array_op = None
        self.prediction   = None
        self.logits       = None
        self.root_logits  = None
        self.root_predict = None

        self.saver = None
        self.best_saver = None

        self.root_loss = None
        self.full_loss = None

        self.training_op = None
        tf.add_to_collection('hyperparameters/lr', self.config.lr)
        tf.add_to_collection('hyperparameters/l2', self.config.l2)
        tf.add_to_collection('hyperparameters/embed_size', self.config.embed_size)
        tf.add_to_collection('hyperparameters/label_size', self.config.label_size)
        #tensor_array_op is the operation on the TensorArray

    # private functions used to construct the graph.
    def _embed_word(self, word_index):
        with tf.variable_scope("Composition", reuse=True) as scope:
            embedding = tf.get_variable("embedding")
        return tf.expand_dims(tf.gather(embedding, word_index), 0)

    # private functions used to construct the graph.
    def _combine_children(self, tensor_concat, left_idx, right_idx):
        left_tensor = tf.expand_dims(tf.gather(tensor_concat, left_idx), 0)
        right_tensor = tf.expand_dims(tf.gather(tensor_concat, right_idx), 0)
        with tf.variable_scope('Composition', reuse=True):
            W1 = tf.get_variable('W1')
            b1 = tf.get_variable('b1')
        return tf.nn.relu(tf.matmul(tf.concat(1, [left_tensor, right_tensor]), W1) + b1)

    def _loop_over_tree(self, i, tensor_list):
        is_leaf = tf.gather(self.is_a_leaf, i)
        word_idx    = tf.gather(self.word_index, i)
        left_child  = tf.gather(self.left_child, i)
        right_child = tf.gather(self.right_child, i)
        node_tensor = tf.cond(is_leaf, lambda : self._embed_word(word_idx),
                                       lambda : self._combine_children(tensor_list, left_child, right_child))
        tensor_list = tf.concat(0, [tensor_list, node_tensor])
        i = tf.add(i,1)
        return i, tensor_list

    # i is the index (over data stored in the placeholders)
    # identical type[out] = type[in]; can be used in while_loop
    # so initial case iteration -> puts left most leaf on the tensorarray (and increments i)
    # next iteration -> puts next left most (leaf on stack) and increments i
    # ....
    # until all the leaves are on the stack in the correct order
    # starts combining the leaves after and adding to the stack
    def construct_tensor_array(self):
        loop_condition = lambda i, tensor_array: \
                         tf.less(i, tf.squeeze(tf.shape(self.is_a_leaf)))
                         #tf.squeeze(tf.shape(placeholder)) <--> length of the storage of all leaves
        
        left_most_element = self._embed_word(tf.gather(self.word_index, 0))
        #index is 1
        i1 = tf.constant(1, dtype=tf.int32)

        while_loop_op = tf.while_loop(cond=loop_condition,
                                       body=self._loop_over_tree,
                                       loop_vars=[i1, left_most_element],
                                       shape_invariants=[i1.get_shape(), tf.TensorShape([None,self.config.embed_size])])
        
        return while_loop_op[1]

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
            embedding = tf.get_variable("embedding",
                                        [self.vocab.total_words, self.config.embed_size])
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
            self.logits = self.add_projections_op(self.tensor_array_op)
        return self.logits

    def root_logits_op(self):
        #construct once
        if self.root_logits is None:
            root_node = tf.expand_dims(self.tensor_array_op[-1,:],0)
            self.root_logits = self.add_projections_op(root_node)
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
            l2_loss = self.l2_reg  * tf.add_n(tf.get_collection("l2_loss"))
            idx = tf.where(tf.less(self.labelholder,2))
            logits = tf.gather(logits, idx)
            labels = tf.gather(labels, idx)
            objective_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            loss = objective_loss + l2_loss
            tf.summary.scalar(name="loss_l2", tensor=l2_loss)
            tf.summary.scalar(name="loss_objective", tensor=tf.reduce_sum(objective_loss))
            tf.summary.scalar(name="loss_total", tensor=loss)
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
            l2_loss = self.l2_reg * tf.add_n(tf.get_collection("l2_loss"))
            objective_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            loss = objective_loss + l2_loss
            tf.summary.scalar(name="root_loss_l2", tensor=l2_loss)
            tf.summary.scalar(name="root_loss_objective", tensor=tf.reduce_sum(objective_loss))
            tf.summary.scalar(name="root_loss_total", tensor=loss)
            self.root_loss = loss
        # END YOUR CODE
        return self.root_loss

    def get_saver(self):
        if self.saver is None:
            print("Creating Saver;")
            self.saver = tf.train.Saver()
        return self.saver

    def get_best_saver(self):
        if self.best_saver is None:
            print("Creating Best Saver (keeps only one checkpoint);")
            self.best_saver = tf.train.Saver(max_to_keep=1, name="best_saver")
        return self.best_saver


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
            optimizer = tf.train.AdamOptimizer(self.learning_rate)#tf.train.GradientDescentOptimizer(self.config.lr)
            #optimizer = tf.train.AdamOptimizer(self.config.lr)
            tf.summary.scalar("lr", self.learning_rate)
            self.training_op = optimizer.minimize(loss, global_step=global_step)
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

    def build_feed_dict(self, in_node, is_training=True):
        nodes_list = []
        tr.leftTraverse(in_node, lambda node, args: args.append(node), nodes_list)

        if is_training:
            feed_dict = {
              self.is_a_leaf   : [ n.isLeaf for n in nodes_list ],
              self.left_child : [ nodes_list.index(n.left) if not n.isLeaf else -1 for n in nodes_list ],
              self.right_child : [ nodes_list.index(n.right) if not n.isLeaf else -1 for n in nodes_list ],
              self.word_index  : [ self.vocab.encode(n.word) if n.word else -1 for n in nodes_list ],
              self.labelholder : [ n.label for n in nodes_list ],
              self.learning_rate : self.config.lr,
              self.l2_reg : self.config.l2
            }
        else:
            feed_dict = {
              self.is_a_leaf   : [ n.isLeaf for n in nodes_list ],
              self.left_child : [ nodes_list.index(n.left) if not n.isLeaf else -1 for n in nodes_list ],
              self.right_child : [ nodes_list.index(n.right) if not n.isLeaf else -1 for n in nodes_list ],
              self.word_index  : [ self.vocab.encode(n.word) if n.word else -1 for n in nodes_list ],
              self.labelholder : [ n.label for n in nodes_list ],
              self.learning_rate : self.config.lr,
              self.l2_reg : 0.
            }
        return feed_dict

    def predict(self, trees, sess, load_weights=False, get_loss=False, is_training=True):
        """Make predictions from the provided model."""
        results = []
        losses = []
        if load_weights is False:
            print("using current session weights for prediction")
        else:
            print("Loading weights from (best weights);")
            best_saver = self.get_best_saver()
            ckpt = tf.train.get_checkpoint_state('./weights/best')
            if ckpt and ckpt.model_checkpoint_path:
                #print(best_saver.last_checkpoints[-1])
                best_saver.restore(sess, best_saver.last_checkpoints[-1])
                #print(ckpt.model_checkpoint_path)
                #print(tf.report_uninitialized_variables(tf.global_variables()))

        logits = self.root_logits_op()
        #evaluation is based upon the root node
        root_loss = self.loss_op(logits=logits, labels=self.labelholder[-1:])
        root_prediction_op = self.root_prediction_op()

        for t in trees:
            feed_dict = self.build_feed_dict(t.root, is_training)
            if get_loss:
                root_prediction, loss = sess.run([root_prediction_op, root_loss], feed_dict=feed_dict)
                losses.append(loss)
                results.append(root_prediction)
            else:
                root_prediction = sess.run(root_prediction_op, feed_dict=feed_dict)
                results.append(root_prediction)
        return results, losses

    #need to rework this: (OP creation needs to be made independent of using OPs)
    def run_epoch(self, sess, summary_writer, new_model=False, verbose=True, epoch=0):
        loss_history = []
        random.shuffle(self.train_data)
        saver = self.get_saver()

        add_model_op = self.add_model()
        logits = self.logits_op()
        loss = self.full_loss_op(logits=logits, labels=self.labelholder)
        train_op = self.training(loss)

        if new_model:
            init = tf.global_variables_initializer()
            self.merged_summaries = tf.summary.merge_all()
            sess.run(init)
        # else:
        #     #
        #     ckpt = tf.train.get_checkpoint_state('./weights')
        #     if ckpt and ckpt.model_checkpoint_path:
        #         saver.restore(sess, ckpt.model_checkpoint_path)
        #         print(tf.report_uninitialized_variables(tf.global_variables()))
        #         #sess.run(tf.variable_initializer(tf.report_uninitialized_variables(tf.all_variables())))

        for step, tree in enumerate(self.train_data):
            feed_dict = self.build_feed_dict(tree.root, is_training=True)
            loss_value, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            merged, current_step = sess.run([self.merged_summaries, global_step], feed_dict=feed_dict)
            summary_writer.add_summary(merged, global_step=current_step)
            loss_history.append(loss_value)
            if verbose:
                sys.stdout.write('\r{} / {} :    loss = {}'.format(
                        step+1, len(self.train_data), np.mean(loss_history)))
                sys.stdout.flush()

        if not os.path.exists("./weights"):
            os.makedirs("./weights")

        #print('./weights/%s.temp'%self.config.model_name)
        print("\nSaving %s"%self.config.model_name)
        out_file = saver.save(sess, './weights/%s.cpkt'%self.config.model_name, global_step=global_step)
        print("File out: ", out_file)
        #print(saver.last_checkpoints)
        train_preds, _ = self.predict(self.train_data, sess)
        val_preds, val_losses = self.predict(self.dev_data, sess, get_loss=True, is_training=False)
        train_labels = [t.root.label for t in self.train_data]
        val_labels = [t.root.label for t in self.dev_data]
        train_acc = np.equal(train_preds, train_labels).mean()
        val_acc = np.equal(val_preds, val_labels).mean()
        print()
        print('Training acc (only root node): {}'.format(train_acc))
        print('Validation acc (only root node): {}'.format(val_acc))
        print(self.make_conf(train_labels, train_preds))
        print(self.make_conf(val_labels, val_preds))
        return train_acc, val_acc, loss_history, np.mean(val_losses)

    def train(self, sess, verbose=True):
        best_saver = self.get_best_saver()

        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = self.config.max_epochs
        #default stop location

        #probably can remove initialization to here
        summary_writer = tf.summary.FileWriter('rnn_logs/test_log/', sess.graph)

        for epoch in range(self.config.max_epochs):
            print('epoch %d'%epoch)
            if epoch==0:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(new_model=True, epoch=epoch, sess=sess, verbose=verbose, summary_writer=summary_writer)
            else:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(epoch=epoch, sess=sess, verbose=verbose, summary_writer=summary_writer)
            complete_loss_history.extend(loss_history)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            #lr annealing
            epoch_loss = np.mean(loss_history)
            if epoch_loss > prev_epoch_loss*self.config.anneal_threshold:
                self.config.lr/=self.config.anneal_by
                print('annealed lr to %f'%self.config.lr)
            prev_epoch_loss = epoch_loss

            #save if model has improved on val
            print("validation loss: %f; prior_best: %f (Epoch %d)" % (val_loss, best_val_loss, best_val_epoch))
            if val_loss < best_val_loss:
                if not os.path.exists("./weights/best"):
                    os.makedirs("./weights/best")
                best_saver.save(sess, './weights/best/%s.cpkt'%(self.config.model_name), global_step=global_step)
                print("saving new (best) checkpoint; (Epoch %d) \nFile: " % epoch, best_saver.last_checkpoints[-1])
                best_val_loss = val_loss
                best_val_epoch = epoch

            # if model has not imprvoved for a while stop
            if epoch - best_val_epoch > self.config.early_stopping:
                stopped = epoch
                break
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
    sess = tf.Session()
    stats = model.train(verbose=True, sess=sess)
    print('Training time: {}'.format(time.time() - start_time))

    plt.figure()
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss_history.png")

    print('Test')
    print('=-=-=')
    predictions_test, _ = model.predict(model.test_data, sess, load_weights=True)
    predictions_dev, _ = model.predict(model.dev_data, sess, load_weights=True)
    labels_test = [t.root.label for t in model.test_data ]
    labels_dev  = [t.root.label for t in model.dev_data ]
    test_acc = np.equal(predictions_test, labels_test).mean()
    dev_acc = np.equal(predictions_dev, labels_dev).mean()
    print('Test acc: {}'.format(test_acc))
    print('Dev  acc: {}'.format(dev_acc))

if __name__ == "__main__":
    test_RNN()
