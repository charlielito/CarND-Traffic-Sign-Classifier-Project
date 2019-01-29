# **Traffic Sign Recognition** 
**Carlos Andres Alvarez Restrepo**
**Self driving nanodegree course**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/test_images.png "tests images"
[image3]: ./examples/after_aug.png "Before augmentation"
[image4]: ./examples/pre_aug.png "Pre augmentation"
[image5]: ./test_images/test0.jpg "Traffic Sign 1"
[image6]: ./test_images/test1.jpg "Traffic Sign 2"
[image7]: ./test_images/test2.jpg "Traffic Sign 3"
[image8]: ./test_images/test3.jpg "Traffic Sign 4"
[image9]: ./test_images/test4.jpeg "Traffic Sign 5"
[image10]: ./test_images/test5.jpg "Traffic Sign 6"
[image11]: ./test_images/test6.jpg "Traffic Sign 7"
[image12]: ./test_images/test7.jpg "Traffic Sign 8"
[image13]: ./test_images/test8.jpg "Traffic Sign 9"

<img src="/test_images/test0.jpg " width="200" height="200" />

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/charlielito/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Summary of the data set

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset

Here is an exploratory visualization of the data set. Here you can see 20 random images with their corresponding class from the training set. As you can see they vary a lot in terms of ilumination and background.

![alt text][image2]

Also a bar chart shows how the data is distributed along the 43 classes. As it can be seen, the data set is quite unbalanced since for example for classes 1 and 2 there are almost 2000 examples whilst for the class 0 there are only approx 250 examples, almost 10x less!


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing of images

The images stayed as color images since some of the colors of the traffic signs may help the network to classify them correctly, i.e. stop signs are red, the border of some signs are also red, and so on. There was also no normalization made since the first experiments showed no much improvement when normalizing. 

The only tecnique used in this stage was to perform a series of transformation to the images in train time to help the network not to overfit the train set. The data augmentation (transformation) pipeline consists in:

* Gaussian Blur: random sigma for blur between 0.0 and 1.0
* Crop: Random pixel crop in all directions from 0 to 6 pixels
* Contrast Normalization: Random values for contrast between 0.7 and 1.3
* Chanels multiplication: Random value between 1.0 and 1.5
* Additive Poisson Noise: Random parameter from 2 to 12
* Random transforms:
    * Random scalization in `x` and `y` direction: random values between 0.9 and 1.1
    * Random translation in both direction: values between -0.1 and 0.1
    * Random rotation between -15 and 15 degrees
    * Random shear between -2 and 2 degrees

The order of each transformation is done randomly. At train time only the 60% of the images are "augmented" randomly.

Here is an example of a traffic sign image before data augmentation:

![alt text][image4]

Here you can see 64 images randomly augmented with the pipeline described:

![alt text][image3]



#### 2. Final model architecture

My final model consisted of the following architecture:

* Inputs: 3 channels 32 x 32 (RGB)
* Convolutional layer: 16 filters, kernel 5x5, padding 'same', RELU activation. Output 32x32x16
* Convolutional layer: 32 filters, kernel 3x3, padding 'same', RELU activation. Output: 32x32x32
* Max Pool: kernel 2x2, stride 2. Output: 16x16x32
* Convolutional layer: 64 filters, kernel 3x3, padding 'same', RELU activation. Output: 16x16x64
* Max Pool: kernel 2x2, stride 2. Output: 8x8x64
* Convolutional layer: 64 filters, kernel 3x3, padding 'same', RELU activation. Output: 8x8x32
* Flatten output -> 4096
* Fully connected: 2048 neurons, RELU activation, dropout = 0.6
* Fully connected: 512 neurons, RELU activation, dropout = 0.6
* Dense layer for output: 43 neurons, Softmax activation
 


#### 3. Training

The optimizer 

The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Iteration 
Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. New German traffic signs found on the web 
Here are 9 German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12]

The first image might be difficult to classify because ...

#### 2. Model's predictions
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Model Certainty 
Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


