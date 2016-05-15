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
