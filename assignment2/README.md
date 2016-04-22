[CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
====================================================================================

Assignment #2: Deep and Recurrent Neural Networks
-------------------------------------------------

**Due Date: 5/5/2016 (Thu) 11:59 PM PST.**

In this assignment you will learn how to use TensorFlow to solve problems in NLP. In particular, you'll use TensorFlow to implement feed-forward neural networks and recurrent neural networks (RNNs), and apply them to the tasks of Named Entity Recognition (NER) and Language Modeling (LM).

As with Assignment #1, you're limited to a maximum of three late days on this assigment. Don't forget that the in-class midterm is scheduled for May 10, so we recommend starting this one early!

Setup
-----

**Note:** *Please be sure you have Python 2.7.x installed on your system. The following instructions should work on Mac or Linux. If you have any trouble getting set up, please come to office hours and the TAs will be happy to help.*

**Get the code (updated!):** [**Download the starter code here**](http://cs224d.stanford.edu/assignment2/assignment2.zip) and [**the assignment handout here**](http://cs224d.stanford.edu/assignment2/assignment2.pdf).

**Python package requirements:** The core requirements for this assignment are
* tensorflow
* numpy

If you have a recent linux (**Ubuntu 14.04** and later) install or Mac OS X, the default TensorFlow installation directions will work well for you. If not, we recommend using the installation on the [**corn clusters**](https://web.stanford.edu/group/farmshare/cgi-bin/wiki/index.php/Main_Page). Note that you will need to use the system default python, not a local Anaconda python.

The corn clusters don't provide GPU support. If you'd like to use GPUs, we recommend using AWS. We've put together a [**brief tutorial**](http://cs224d.stanford.edu/supplementary/aws-tutorial-2.pdf) with directions on how to get started with TensorFlow on AWS.

Submitting your work
--------------------

Once you are done working, run the `collectSubmission.sh` script; this will produce a file called `assignment2.zip`. Rename this file to `<your-sunet-id>.zip`, for instance if your stanford email is `jdoe@stanford.edu`, your file name should be `jdoe.zip`.

For the written component, please upload a PDF file of your solutions to `Gradescope`. If you are enrolled in the class you should have been signed up automatically. If you added the class late or are not signed up, post privately to Piazza and we will add you to the roster. When asked to map question parts to your PDF, please map the parts accordingly as courtesy to your TAs. This is crucial so that we can provide accurate feedback. If a question has no written component (completely programatic), map it on the same page as the previous section or next section.

Please upload your programming submission below.


Assignment Overview (Tasks)
---------------------------

There will be three parts to this assignment. Each part has written and code components. The assignment is designed to be completed in order as later sections will leverage solutions to earlier parts. We recommend reading the assignment carefully and starting early as some parts may take significant time to run.

Q1: TensorFlow Softmax (20 points)
----------------------------------

Q2: TensorFlow NER Window Model (35 points)
-------------------------------------------

Q3: TensorFlow RNN Language Model (45 points)
---------------------------------------------
