# FeedForward-Neural-Network

## Abstract

In this assignment, we will be performing feed-forward neural networks in pytorch. The first part of the assignment is using the neural network for curve fitting a set of data. The second part of the assignment is classifying digits from the MNIST dataset using the neural network. We will be comparing the results with other types of classification systems like LSTM, SVD, and DTC. 

## Introduction

The structure of the neural network allows lots of freedom in the variables that it requires. The assignment introduces the process of building this type of neural network. For the first part of the assignment, I built a neural network that can curve fit a set of data points with different scenarios. Second, I designed another neural network to classify the MNIST data set and reduce the dimensionality by PCA for 20 modes. Afterward, we will compare the results with other types of classifiers like LSTM, SVD, and DTC. 

## Theoretical Background

The feedforward neural network is a process in which we input a set of values into layers of neurons and the input changes depending on the functions and changes of weights to create our desired output. What we want is for the input to go through the layers of hidden layers so that the output meets our goals. The way to change the input is to have a layer that has an activation function. We can also change the number of neurons in each layer. Afterward, we can set up the learning rate of the neural network. The learning rate acts like the stepsize of the neural network. If we are dealing with a large amount of data, then setting up a batch size of the training and testing loaders can help with the speed of the neural network. With all these variables, the ability to test and build the neural network takes time and patients so that we can reach the goals.

## Algorithm, Implementation, and Development

To create the architecture of the neural network, we must set the number of layers of the neural network. For the first part, we have three layers of the feedforward neural network. Each layer is set as linear. The neuron sequence is 1 to 10 to 5 to 1. The activation function for the first two layers is the ReLU function. Then, the last layer is a linear activation function.

Next, we need to set up the loss function and optimizer. The criterion is set up to use the mean square error loss function and the optimizer is using the Adam with a learning rate of 0.01. The Adam optimizer is best used for regression problems like the curve fitting problem. 

Afterward, we need to create the epoch training for the neural network. The number of epochs is the number of steps that we are taking. In this case, we will be doing 1000 epochs. For the first part of the assignment, we need the first 20 data to be training data and the last 10 should be the testing data.  Then, we input the training data into the model and the loss function takes in the output of the model and the labels from the training data. Afterward, we print out the loss of the training data and the loss of the testing data. We will be doing the same process but for the first 10 and last 10 data sets as the training data and the middle 10 as the testing data. For visualization, I plotted the model over each epoch step to show the model slowly learning how to fit a curve to the model. 

Next, we will need to perform PCA on the MNIST dataset with 20 modes for the next portion of the assignment. Since the data set is large, we have the training and testing loaders have a batch size of 64 The neural network has three layers with all being linear sequences but the first layer starts with 20 neurons as the input layer. Then, the neurons will go to 128 neurons and 64 neurons until it reaches 10 outputs. For the activation function, we have to flatten the input images and start the activation functions with two ReLU layers and one linear function. 

Then, the criterion of the function is cross-entropy loss and the SGD for the optimizer. These functions are great for using classifier neural networks. The learning stepsize is 0.01. 

The number of epochs is set to 5 for the training portion of the neural network. The loss is calculated by comparing the output of inputting the images into the model and the labels for the images. Afterward, we can calculate the testing accuracy of the model by finding the number of correct labels and the total number of images. 

The assignment asks us to compare the resulting outputs for the MNIST feedforward neural network with LSTM, SVD, and DTC. The SVD and DTC are similar to homework assignment 3 where we compare classifiers. Therefore, the code is similar to the code from that assignment. The LSTM is a different type of neural network where we use the LSTM code in the neural network architecture. 

## Computational Results

From Figure 1, the training data vs the model predictions show that the model slowly transforms from a linear near 0 to a linear line that matches the training data. Therefore, the loss of the training data is around 5 and the testing data is around 33. However, the model prediction is not that great in fitting the testing data. For comparison, in homework 1, we designed a similar curve fit using the polyfit function. The least-square error from that assignment was 2.24 for the training data and 3.49 for the testing data. The training data is similar in value and may be decreased in our neural network if we would increase the number of epochs since the graphs show that the curve is slowly fitting the dataset. However, the testing least square error is shown to have a huge difference in value. The reason is that the model is slowly overfitting the training set but does not account for the testing set. In Figure 2, we can see the testing and training loss for the entire epoch. We can see a similar result with the training model that takes in the first and last 10 data sets as the training data and the middle 10 as the testing data.  The least-square error after 1000 epochs is around 3 for the training set and 7.87 for the testing set. Compared with the first homework, the first homework assignment has around 2 for the training set and 3 for the testing set if we account for the linear function in the homework assignment. In Figure 3, we can see the model being formed similar to Figure 1.  Figure 4 shows every 10 losses for every 10 epoch steps. 

Taking a look at the next part of the assignment, we can see that the neural network for the MNIST classifier shows the result of decreasing the features with the 20 PCA modes. After ten epochs, the accuracy score of the classifier for the feedforward neural network is around **93%**. Comparing it to the LSTM neural network, we can see that the accuracy score is **96.57%**. The accuracy score for the other two classifiers, SVD and DTC, is **99%** and **85%**. We can see that if we wanted to reduce the dimensionality of the data then it is better to use the SVD and LSTM classifiers than the DTC and FFNN.  

## Conclusion

The resulting accuracy scores and loss show that the feedforward neural network is better when we get more data. For example, the neural network does not perform well when given a smaller dataset while increasing the number of data points increases the accuracy of the neural network. 

Also, the feedforward neural network shows that the accuracy score of a reduced dimensionality dataset is around 93%, but the SVD is the best-performing classifier while the LSTM is the best-performing neural network classifier. Therefore, the feedforward neural network is best used when given all of the data points but if we wanted to reduce the dimensionality of the dataset then using SVD or LSTM is the best option.  

