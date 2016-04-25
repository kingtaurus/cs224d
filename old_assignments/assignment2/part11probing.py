#!/usr/bin/env python

import sys, os
from numpy import *

def print_scores(scores, words):
    for i in range(len(scores)):
        print "[%d]: (%.03f) %s" % (i, scores[i], words[i])


def part_a(clf, num_to_word, verbose=True):
    """
    Code for 1.1 part (a):
    Hidden Layer, Center Word

    clf: instance of WindowMLP,
            trained on data
    num_to_word: dict {int:string}

    You need to create:
    - topscores : list of lists of 10 scores (float)
    - topwords  : list of lists of 10 words (string)
    You should generate these lists for each neuron
    (so for hdim = 100, you'll have lists of 100 lists of 10)
    then fill in neurons = [<your chosen neurons>] to print
    """
    #### YOUR CODE HERE ####





    neurons = [1,3,4,6,8] # change this to your chosen neurons

    #### END YOUR CODE ####
    # topscores[i]: list of floats
    # topwords[i]: list of words
    if verbose == True:
        for i in neurons:
            print "Neuron %d" % i
            print_scores(topscores[i], topwords[i])

    return topscores, topwords


def part_b(clf, num_to_word, num_to_tag, verbose=True):
    """
    Code for 1.1 part (b):
    Model Output, Center Word

    clf: instance of WindowMLP,
            trained on data
    num_to_word: dict {int:string}

    You need to create:
    - topscores : list of 5 lists of 10 probability scores (float)
    - topwords  : list of 5 lists of 10 words (string)
    where indices 0,1,2,3,4 correspond to num_to_tag, i.e.
    tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
    """
    #### YOUR CODE HERE ####






    #### END YOUR CODE ####
    # topscores[i]: list of floats
    # topwords[i]: list of words
    if verbose == True:
        for i in range(1,5):
            print "Output neuron %d: %s" % (i, num_to_tag[i])
            print_scores(topscores[i], topwords[i])
            print ""

    return topscores, topwords


def part_c(clf, num_to_word, num_to_tag, verbose=True):
    """
    Code for 1.1 part (c):
    Model Output, Preceding Word

    clf: instance of WindowMLP,
            trained on data
    num_to_word: dict {int:string}

    You need to create:
    - topscores : list of 5 lists of 10 probability scores (float)
    - topwords  : list of 5 lists of 10 words (string)
    where indices 0,1,2,3,4 correspond to num_to_tag, i.e.
    tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
    """
    #### YOUR CODE HERE ####






    #### END YOUR CODE ####
    # topscores[i]: list of floats
    # topwords[i]: list of words
    if verbose == True:
        for i in range(1,5):
            print "Output neuron %d: %s" % (i, num_to_tag[i])
            print_scores(topscores[i], topwords[i])
            print ""

    return topscores, topwords


##
# Dummy test code
# run this script, and make sure nothing crashes
# (this is the same as sanity check for part 1.1)
if __name__ == '__main__':
    num_to_word = dict(enumerate(
                       ["hello", "world", "i", "am", "a", "banana",
                        "there", "is", "no", "spoon"]))
    tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
    num_to_tag = dict(enumerate(tagnames))

    from nerwindow import WindowMLP
    random.seed(10)
    wv = random.randn(10,50)
    clf = WindowMLP(wv, windowsize=3,
                    dims = [None, 100, 5], rseed=10)

    print "\n=== Testing Part (a) ===\n"
    s,w = part_a(clf, num_to_word, verbose=True)
    assert(len(s) == len(w))
    if type(s) == dict: # some students may have done this
        for k in s.keys(): assert(k in w)
        for k in w.keys(): assert(k in s)
        assert(len(s) >= 5)
    else: # list
        assert(len(s[0]) == len(w[0]))
        assert(len(s[0]) == 10)
        assert(type(w[0][0]) == str)

    print "\n=== Testing Part (b) ===\n"
    s,w = part_b(clf, num_to_word, num_to_tag, verbose=True)
    assert(len(s) == len(w))
    assert(len(s) == 5)
    assert(len(s[0]) == len(w[0]))
    assert(len(s[0]) == 10)
    assert(type(w[0][0]) == str)

    print "\n=== Testing Part (c) ===\n"
    s,w = part_c(clf, num_to_word, num_to_tag, verbose=True)
    assert(len(s) == len(w))
    assert(len(s) == 5)
    assert(len(s[0]) == len(w[0]))
    assert(len(s[0]) == 10)
    assert(type(w[0][0]) == str)