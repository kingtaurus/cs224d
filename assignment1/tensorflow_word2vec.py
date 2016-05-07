import os
import math
import random
import collections

import numpy as np
import tensorflow as tf

import cs224d.data_utils as data_utils
from tensorflow.models.embedding import gen_word2vec as word2vec

class Options(object):
    def __init__(self):
        #Model Options
        self.emb_dim = 20
        self.train_data  = None
        self.num_samples = 20
        self.learning_rate = 1.0

        self.epochs_to_train = 5
        self.batch_size  = 64
        self.window_size = 5
        self.min_count   = 3

class Word2Vec(object):
    """Word2Vec model (skipgram) """
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()
        self.build_eval_graph()
        self.save_vocab()
        self._read_dataset()

    def _read_dataset(self):
        # dataset = data_utils.StanfordSentiment()
        # #print(dataset.sent_labels()[0:100])
        # #print(dataset.getSplitSentences(0)[0:100])
        # #this is the labels vector :)

        # #sentences = np.from_iter(dataset.sentences(), dtype="int32")
        # self._word2id = dataset.tokens()
        # print(self._word2id["UNK"])
        # ids = [self._word2id.get(w) for w in self._word2id.keys()]
        # print(ids)
        pass
    def forward(self, examples, labels):
        return None,None

    def nce_loss(self, true_logits, sampled_logits):
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(true_logits, tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(sampled_logits, tf.zeros_like(sampled_logits))
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor

    def build_graph(self):
        opts = self._options
        (words, counts, words_per_epoch, self._epoch, self._words, examples,
        labels) = word2vec.skipgram(filename="text8",
                          batch_size=opt.batch_size,
                          window_size=opt.window_size,
                          min_count=opt.min_count,
                          subsample=0)
        (opts.vocab_words, opts.vocab_counts,
         opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)
        self._examples = examples
        self._labels = labels
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i
        true_logits, sampled_logits = self.forward(examples, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.scalar_summary("NCE loss", loss)
        self._loss = loss
        self.optimize(loss)

    def build_eval_graph(self):
        pass
    def save_vocab(self):
        pass

if __name__ == "__main__":
    opt     = Options()
    session = tf.Session()
    model = Word2Vec(opt, session)
