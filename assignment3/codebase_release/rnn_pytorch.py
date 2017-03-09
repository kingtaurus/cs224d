import sys
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm

import tree as tr
from utils import Vocab

from collections import OrderedDict

import seaborn as sns

from random import shuffle

sns.set_style('whitegrid')

embed_size = 100
label_size = 2
early_stopping = 2
anneal_threshold = 0.99
anneal_by = 1.5
max_epochs = 30
lr = 0.01
l2 = 0.02
average_over = 700
train_size = 800


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

  def init_variables(self):
    print("total_words = ", self.vocab.total_words)

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

  def forward(self, x):
    """
    Forward function accepts input data and returns a Variable of output data
    """
    self.node_list = []
    root_node = self.walk_tree(x.root)
    all_nodes = torch.cat(self.node_list)
    #now I need to project out
    return all_nodes

def main():
  print("do nothing")


if __name__ == '__main__':
  train_data, dev_data, test_data = tr.simplified_data(train_size, 100, 200)
  vocab = Vocab()
  train_sents = [t.get_words() for t in train_data]
  vocab.construct(list(itertools.chain.from_iterable(train_sents)))
  model   = RNN_Model(vocab, embed_size=50)
  main()

  lr = 0.01
  loss_history = []
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.0)
  # params (iterable): iterable of parameters to optimize or dicts defining
  #     parameter groups
  # lr (float): learning rate
  # momentum (float, optional): momentum factor (default: 0)
  # weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
  #torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0)
  # print(model.fcl._parameters['weight'])

  for epoch in range(max_epochs):
    print("epoch = ", epoch)
    shuffle(train_data)
    total_root_prediction = 0.
    total_summed_accuracy = 0.
    if (epoch % 10 == 0) and epoch > 0:
        for param_group in optimizer.param_groups:
          #update learning rate
          print("Droping learning from %f to %f"%(param_group['lr'], 0.5 * param_group['lr']))
          param_group['lr'] = 0.5 * param_group['lr']
    for step, tree in enumerate(train_data):
        # if step == 0:
        #   optimizer.zero_grad()
        # objective_loss.backward()
        # if step == len(train_data) - 1:
        #   optimizer.step()

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

      correct = predictions.data == torch_labels
      #so correctly predicted (root);
      total_root_prediction += float(correct[-1])
      total_summed_accuracy += float(correct.sum()) / len(labels)

      objective_loss = F.cross_entropy(input=logits_squeezed, target=Variable(torch_labels))
      if objective_loss.data[0] > 5 and epoch > 10:
        #interested in phrase that have large loss (i.e. incorrectly classified)
        print(' '.join(tree.get_words()))

      loss_history.append(objective_loss.data[0])
      if step % 20 == 0 and step > 0:
        print("step %3d, last loss %0.3f, mean loss (%d steps) %0.3f" % (step, objective_loss.data[0], average_over, np.mean(loss_history[-average_over:])))
      optimizer.zero_grad()

      if np.isnan(objective_loss.data[0]):
        print("object_loss was not a number")
        sys.exit(1)
      else:
        objective_loss.backward()
        clip_grad_norm(model.parameters(), 5, norm_type=2.)
        #temp_grad += model.fcl._parameters['weight'].grad.data
        # # Update weights using gradient descent; w1.data and w2.data are Tensors,
        # # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
        # # Tensors.
        # loss.backward()
        # w1.data -= learning_rate * w1.grad.data
        # w2.data -= learning_rate * w2.grad.data
        optimizer.step()
    print("total root predicted correctly = ", total_root_prediction/ float(train_size))
    print("total node (including root) predicted correctly = ", total_summed_accuracy / float(train_size))

    total_dev_loss = 0.
    dev_correct_at_root = 0.
    dev_correct_all = 0.
    for step, dev_example in enumerate(dev_data):
      all_nodes = model(dev_example)

      labels  = []
      indices = []
      for x,y in enumerate(dev_example.labels):
        if y != 2:
          labels.append(y)
          indices.append(x)
      torch_labels = torch.LongTensor([l for l in labels if l != 2])
      logits = all_nodes.index_select(dim=0, index=Variable(torch.LongTensor(indices)))
      logits_squeezed = logits.squeeze()
      predictions = logits.max(dim=2)[1].squeeze()

      correct = predictions.data == torch_labels
      #so correctly predicted (root);
      dev_correct_at_root += float(correct[-1])
      dev_correct_all += float(correct.sum()) / len(labels)
      objective_loss = F.cross_entropy(input=logits_squeezed, target=Variable(torch_labels))
      total_dev_loss += objective_loss.data[0]
    print("total_dev_loss = ", total_dev_loss)
    print("correct (root) = ", dev_correct_at_root)
    print("correct (all)= ", dev_correct_all)
  # logits = logits.index_select(dim=0, index=Variable(torch.LongTensor(indices)))
  plt.figure()
  plt.plot(loss_history)
  plt.show()
  print("DONE!")