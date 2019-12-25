# logistics-regression-with-iris_dataset

This is the project of numerical analysis class

## 1. Introduction

In this project, The problem is to classify the iris flower. There are two different classes of iris flower (setosa and versicolor)

To model this problem, I use a logistic regression model, which has following formulation:

$y = \frac{1}{e^{-z} + 1}$ and $z = W^TX+b$

In this problem, each sample has four features. z is a linear combination of these features. Then use **sigmoid** function to get the truth probability of this sample.

To solve this model, there are many methods, such as **newton method**, **gradient descent** or other variant.

### 1.1 Gradient Descent

This is a iteration method. First the algorithm initial the weights randomly. There will be a loss function of problem. In this project I use log loss, which means the distance of predicted lable and truth lable. The algorithm is to minimize the loss. Each time it uses training data to get the new loss and find teh gradient. Then it uses gradient to update the weights. Repeat this process until the loss is acceptable or reach the max iteration number.

### 1.2 Stochastic Average Gradient

In this project, I implement with **stochastic average gradient (SAG)** method. The detial of this method can be found [here](<https://papers.nips.cc/paper/4633-a-stochastic-gradient-method-with-an-exponential-convergence-_rate-for-finite-training-sets.pdf>). This method is much faster (converge) than traditional stochastic gradient method. It's a variant of stochastic gradient method.

## 2. Requirement

* python $\geq$ 3.6

* sklearn

* numpy

## 3. Get started

The main process is `mian.py`. To run this mdoel:

`python main.py [--build_in]`

`--build_in` arguments will use the built-in logistic regerssion in skleran package. So you can run this file with and without `build_in` to compare the results. You will find the resluts are the same.
