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
    tf.summary.histogram(name, variable)

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


class RNN_Model():

    def load_data(self):
        """Loads train/dev/test data and builds vocabulary."""
        self.train_data, self.dev_data, self.test_data = tr.simplified_data(700, 100, 200)

        # build vocab from training data
        self.vocab = Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

    def inference(self, tree, predict_only_root=False):
        """For a given tree build the RNN models computation graph up to where it
            may be used for inference.
        Args:
            tree: a Tree object on which to build the computation graph for the RNN
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        node_tensors = self.add_model(tree.root)
        if predict_only_root:
            node_tensors = node_tensors[tree.root]
        else:
            node_tensors = [tensor for node, tensor in node_tensors.items() if node.label!=2]
            node_tensors = tf.concat(0, node_tensors)
        return self.add_projections(node_tensors)

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
        with tf.variable_scope('Composition'):
            ### YOUR CODE HERE
            #initializer=initializer=tf.random_normal_initializer(0,3)
            embedding = tf.get_variable("embedding",
                                        [self.vocab.total_words, self.config.embed_size])
            W1 = tf.get_variable("W1", [2 * self.config.embed_size, self.config.embed_size])
            b1 = tf.get_variable("b1", [1, self.config.embed_size])
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
            ### END YOUR CODE

    def add_model(self, node):
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
        with tf.variable_scope('Composition', reuse=True):
            ### YOUR CODE HERE
            embedding = tf.get_variable("embedding")
            W1 = tf.get_variable("W1")
            b1 = tf.get_variable("b1")
            l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1)
            tf.add_to_collection(name="l2_loss", value=l2_loss)
            ### END YOUR CODE

        W_split = tf.split(0, 2, W1)
        W_left  = W_split[0]
        W_right = W_split[1]
        node_tensors = OrderedDict()
        curr_node_tensor = None
        if node.isLeaf:
            ### YOUR CODE HERE
            word_id = self.vocab.encode(node.word)
            curr_node_tensor = tf.expand_dims(tf.gather(embedding, word_id),0)
            ### END YOUR CODE
        else:
            node_tensors.update(self.add_model(node.left))
            node_tensors.update(self.add_model(node.right))
            ### YOUR CODE HERE
            #tf.concat(0,[node_tensors[node.left], node_tensors[node.right]])
            #This operation could be done without the split call above
            #curr_node_tensor = tf.nn.relu(tf.matmul(child_tensor, W1) + b1)
            curr_node_tensor = tf.matmul(node_tensors[node.left], W_left) + tf.matmul(node_tensors[node.right], W_right) + b1
            ### END YOUR CODE
        node_tensors[node] = curr_node_tensor
        return node_tensors

    def add_projections(self, node_tensors):
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

    def loss(self, logits, labels):
        """Adds loss ops to the computational graph.

        Hint: Use sparse_softmax_cross_entropy_with_logits
        Hint: Remember to add l2_loss (see tf.nn.l2_loss)
        Args:
            logits: tensor(num_nodes, output_size)
            labels: python list, len = num_nodes
        Returns:
            loss: tensor 0-D
        """
        loss = None
        # YOUR CODE HERE
        l2_loss = self.config.l2 * tf.get_collection("l2_loss")[0]
        objective_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        loss = objective_loss + l2_loss
        tf.summary.scalar("loss_l2", l2_loss)
        tf.summary.scalar("loss_objective", tf.reduce_sum(objective_loss))
        tf.summary.scalar("loss_total", loss)
        # END YOUR CODE
        return loss

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
        train_op = None
        # YOUR CODE HERE
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        #optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        # END YOUR CODE
        return train_op

    def predictions(self, y):
        """Returns predictions from sparse scores

        Args:
            y: tensor(?, label_size)
        Returns:
            predictions: tensor(?,1)
        """
        predictions = None
        # YOUR CODE HERE
        predictions = tf.argmax(y, dimension=1)
        # END YOUR CODE
        return predictions

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.merged_summaries = None
        self.summary_writer = None

    def predict(self, trees, weights_path, get_loss = False):
        """Make predictions from the provided model."""
        results = []
        losses = []
        for i in range(int(math.ceil(len(trees)/float(RESET_AFTER)))):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                saver = tf.train.Saver()
                saver.restore(sess, weights_path)
                for tree in trees[i*RESET_AFTER: (i+1)*RESET_AFTER]:
                    logits = self.inference(tree, True)
                    predictions = self.predictions(logits)
                    root_prediction = sess.run(predictions)[0]
                    if get_loss:
                        root_label = tree.root.label
                        loss = sess.run(self.loss(logits, [root_label]))
                        losses.append(loss)
                    results.append(root_prediction)
        return results, losses

    def run_epoch(self, new_model = False, verbose=True, epoch=0):
        step = 0
        loss_history = []
        while step < len(self.train_data):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                if new_model:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    new_model = False
                else:
                    saver = tf.train.Saver()
                    saver.restore(sess, './weights/%s.temp'%self.config.model_name)
                for r_step in range(RESET_AFTER):
                    if step>=len(self.train_data):
                        break
                    tree = self.train_data[step]
                    logits = self.inference(tree)
                    labels = [l for l in tree.labels if l!=2]
                    loss = self.loss(logits, labels)
                    train_op = self.training(loss)
                    #initialize_uninitialized_vars(sess)
                    if r_step == 0:
                        self.merged_summaries = tf.summary.merge_all()
                        # self.summary_writer = tf.train.SummaryWriter("tree_rnn_log/", sess.graph)
                    if step == 0 and epoch == 0:
                        self.summary_writer = tf.summary.FileWriter("tree_rnn_log/", sess.graph)
                    loss, _, merged = sess.run([loss, train_op, self.merged_summaries])
                    if step % (RESET_AFTER//2):
                        self.summary_writer.add_summary(merged, epoch * len(self.train_data) + step)
                    loss_history.append(loss)
                    if verbose:
                        sys.stdout.write('\r{} / {} :    loss = {}'.format(
                            step, len(self.train_data), np.mean(loss_history)))
                        sys.stdout.flush()
                    step+=1
                saver = tf.train.Saver()
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                saver.save(sess, './weights/%s.temp'%self.config.model_name, write_meta_graph=False)
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
