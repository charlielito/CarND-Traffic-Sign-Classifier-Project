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
[image5]: ./examples/predictions0.png "Preds"
[image6]: ./examples/predictions1.png "Preds2"
[image7]: ./examples/top5_0.png "Top5 1"
[image8]: ./examples/top5_1.png "Top5 2"
[image9]: ./examples/top5_2.png "Top5 3"
[image10]: ./examples/top5_3.png "Top5 4"
[image11]: ./examples/top5_4.png "Top5 5"
[image12]: ./examples/top5_5.png "Top5 6"
[image13]: ./examples/top5_6.png "Top5 7"
[image14]: ./examples/top5_7.png "Top5 8"



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

Here is an exploratory visualization of the data set. Here you can see 20 random images with their corresponding class from the training set. As you can see they vary a lot in terms of illumination and background.

![alt text][image2]

Also a bar chart shows how the data is distributed along the 43 classes. As it can be seen, the data set is quite unbalanced since for example for classes 1 and 2 there are almost 2000 examples whilst for the class 0 there are only approx 250 examples, almost 10x less!


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing of images

The images stayed as color images since some of the colors of the traffic signs may help the network to classify them correctly, i.e. stop signs are red, the border of some signs are also red, and so on. There was also no normalization made since the first experiments showed no much improvement when normalizing.

The only technique used in this stage was to perform a series of transformation to the images in train time to help the network not to overfit the train set. The data augmentation (transformation) pipeline consists in:

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

To train the model, I use the AdamOptimizer, with a batch size of 128, 250 epochs and a learning rate of 0.001, that was slowing decaying, i.e. each epoch it was multiplied by 0.999.

To avoid saving all models each epoch, I save the model if and only if it had an accuracy greater than 95% on the evaluation set. After that, it was tracked the best model on the eval set and saved only that model.

At the end, the model had an accuracy of 97.2% on the evaluation set and 99,9% in the train set.

#### 4. Iteration

My final model results were:

* training set accuracy of 0.999
* validation set accuracy of 0.972
* test set accuracy of 0.951

The validation accuracy was calculated during training as it can be seen in the Jupyter notebook after each epoch. The training and test accuracy were calculated after that in the "Analyze performance part".

I took a common architecture of a couple of convolutional layers stacked (4 layers), using maxpooling between layers, and 2 fully connected layers at the end before the output layer. The number of kernels in the convolutional part were increasing from 16 to 64 kernels at the last layer. This simple architecture is similar to LeNet and works pretty well but it must be regularized.

At the beginning I tried it without using dropout but the evaluation accuracy was below 0.93. Then I used dropout only in the last fully connected layer, and got above 0.93, but I wanted to get more than 0.97, since with 0.93 seemed that the network still overfitted the train set. That is why I also added dropout to the other fully connected layer, and also performed aggressive data augmentation (the first trys I did not do any pre processing). With this modification I let the model train for long enough to get slightly more than 0.97 in the dev set.

At the end, the dev accuracy and train accuracy were closed to each other, indicating that the model had learned good enough, but of course it could improve more.

### Test a Model on New Images

#### 1. New German traffic signs found on the web
Here are 8 German traffic signs that I found on the web:

<img src="/test_images/test0.jpg " width="200" />
<img src="/test_images/test1.jpg " width="200" />
<img src="/test_images/test2.jpg " width="200" />
<img src="/test_images/test3.jpg " width="200" />
<img src="/test_images/test4.jpeg " width="200" />
<img src="/test_images/test5.jpg " width="200" />
<img src="/test_images/test6.jpg " width="200" />
<img src="/test_images/test7.jpg " width="200" />


The two first images might be difficult to classify because of the background. The third image is interesting because it has no background and no colors, so it may be different from the training data. The 4th image can be difficult because of the watermarks of the image. The 5th also because it has no background at all. The 6th because the sign appears small in comparison with the rest of the image. The 7th because of the red background and the 8th may be not that difficult.

#### 2. Model's predictions


Here are the results of the predictions. It displays the class ID, the probability and also shows an example in the train set of that predicted class.


![alt text][image5] ![alt text][image6]

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5 %. This compares not so good to the accuracy on the test set of 95.1 %. This can be because the different sizes of this web test and the test set. Also, the image where the models gets it wrong has the sign very small so this can be tricky for the network. I made a test where the sign was cropped centrally and the model got it right.


#### 3. Model Certainty

For the first image, the model is very sure that it is a 30 km/h speed limit sign (probability of 1.0), and the image does contain that sign. The top five softmax probabilities can be seen in the bar chart.

![alt text][image7]

For the second image, the model is also very confident that the image is a "go straight or right" sign with 1.0 of confidence. The top five softmax probabilities can be seen in the bar chart.

![alt text][image8]

For the third image, the model is not that sure that this is a "slippery road" sign (probability of 0.29), but the image does contain a "slippery road" sign. The top five softmax probabilities were: slippery road, dangerous curve to the left, double curve, wild animals and right-of-a-way at the next intersection as it can be seen in the next bar chart. It make sense that the model is not that certain since the 3 first top max probabilities classes could correspond to the image (curves), since the image is very pixelated.

![alt text][image9]

For this 4th image, the model is also very confident that the image is a "keep right" sign with 1.0 of confidence. The top five softmax probabilities can be seen in the bar chart.

![alt text][image10]


For the 5th, 6th and 7th images, the model is very confident that the sign are right-of-a-way at the next intersection sign, 20 km/h speed limit sign, and road work sign with 1.0 of confidence. The top five softmax probabilities can be seen in the bar charts with almost no probability (0%).

![alt text][image11]

![alt text][image12]

![alt text][image13]


For this last image, the model predicted a 80km/h speed limit sign with a low probability, i.e. 32.8 %. The top five softmax probabilities correspond all to speed limit sign, being the third max probability the right prediction (approx 14%). This is interesting since the model is not sure of which of the speed limit signs this image belongs to. This can be, again, because the sign is small in comparison with the rest of the image, and in the train set not images like this appear.

![alt text][image14]
