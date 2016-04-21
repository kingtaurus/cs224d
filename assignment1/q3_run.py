import random
import numpy as np
from cs224d.data_utils import *
import matplotlib.pyplot as plt

from q3_word2vec import *
from q3_sgd import *

import seaborn as sns
sns.set(style='whitegrid', context='talk')

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)
# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / \
    dimVectors, np.zeros((nWords, dimVectors))), axis=0)
wordVectors0 = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
        negSamplingCostAndGradient),
    wordVectors, 0.30, 40000, None, True, PRINT_EVERY=10)
print("sanity check: cost at convergence should be around or below 10")

# sum the input and output word vectors
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

# Visualize the word vectors you trained
_, wordVectors0, _ = load_saved_params()
print(wordVectors0.shape)
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"]
visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

plt.figure(figsize=(12,12))
for i in range(len(visualizeWords)):
    plt.scatter(x=coord[i,0], y=coord[i,1])
    plt.text(coord[i,0]+0.01, coord[i,1]+0.01, visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))
plt.xlim((np.min(coord[:,0])-0.1, np.max(coord[:,0])+0.1))
plt.ylim((np.min(coord[:,1])-0.1, np.max(coord[:,1])+0.1))
plt.xlabel("SVD[0]")
plt.ylabel("SVD[1]")

plt.savefig('q3_word_vectors.png')
plt.show()
