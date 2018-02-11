# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a chart showing the distribution of classes in the training set.

![alt text](https://github.com/looboon/CarND-Traffic-Sign-Classifier-Project/blob/master/markup_images/training_distribution.png)

From the graph, we can see that some of the classes have a much higher number of samples in the training dataset compared to the rest, such as speed limit signs for 30, 50kmph. This may make the neural network learn more of these sample and be biased towards learning these few classes compared to the others, especially if the training sample distribution and the actual distribution of the signs differ greatly, resulting in what we call a covariate shift.

We also took a sample of a picture from each class to take a look at the image corresponding to each label.

![alt text](https://github.com/looboon/CarND-Traffic-Sign-Classifier-Project/blob/master/markup_images/exploration.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For image preprocessing, I observed that most importantly, the images need to be normalized so that the image data distribution has 0 mean and equal variance. Without this, the neural network would have a difficult time trying to learn from the image data. We can do this by using a tool in opencv:

cv2.normalize(image, zeros, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

This normalizes our images for our optimizer. We can see the effects of preprocessing before:

![alt text](https://github.com/looboon/CarND-Traffic-Sign-Classifier-Project/blob/master/markup_images/before_preprocessing.png)

And also after the preprocessing:

![alt text](https://github.com/looboon/CarND-Traffic-Sign-Classifier-Project/blob/master/markup_images/after_preprocessing.png)

What we can observe is that the contrast is much sharper between the white regions in the sign and the other symbols in this sign. This preprocessing will definitely help later when we train the convolutional kernels later in the CNN, as it would be much easier to construct convolutional kernels that specialise in detecting these edge featuers if the contrast is much higher.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image  	| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 	|
| Dropout					|	keep probability = 0.5			|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Dropout					|	keep probability = 0.5			|
| Flatten	| 800 output        									|
| Fully connected		| 800 input, 256 output |
| RELU					|												|
| Batch Norm					|												|
| Dropout					|	keep probability = 0.5			|
| Fully connected		| 256 input, 128 output 		|
| RELU					|												|
| Batch Norm					|												|
| Dropout					|	keep probability = 0.5			|
| Softmax				| 128 input, 43 output classes 	|
 
I used a modified version of the LeNet 5 architecture for the traffic sign classifier. This modified architecture has a few differences:

- More convolutional filters at the conv layers: I noticed that performance improved after adding more filters, suggesting that more features need to be learnt for this traffic sign as compared to the original architecture that was trained to classify MNIST
- Dropout: to prevent overfitting to the training data
- Batch Normalization: to prevent internal covariate shift at the fully connected layers

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:

optimizer: Adam
batch size: 128
number of epochs: 30
learning rate: 0.001
dropout rate: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 97.7% 
* test set accuracy of 95.5%

I chose to work on an existing architecture and modify the architecture iteratively along the way.

**What was the first architecture that was tried and why was it chosen?**

I chose the LeNet architecture as a starting point as it was a simple architecture and felt that it would serve the needs for this simple traffic sign classification.

**What were some problems with the initial architecture?**

I noticed that using the initial architecture without doing any changes, the performance could only hit at most more than 80% and could not reach the 93% accuracy target for the validation set. I suspected that this means that the original LeNet architecture alone was not sufficient to tackle the problem of learning to see traffic sign classifiers and had to be modified.

**How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.**

I firstly added more convolutional filters as I suspected that more filters were needed to learn additional features. The LeNet for MNIST probably had to learn much lesser features as the MNIST dataset had only 9 types of handwritten digits, compared to 43 different types of traffic signs for this data.

Another thing to add was dropout as I observed quite a strong case of overfitting on the training set before adding dropout. The difference between the training and validation accuracy shrunk quite abit after adding dropout between successive conv and fully connected layers.

**Which parameters were tuned? How were they adjusted and why?**

Firstly, the batch size and learning rate were tuned. As I noticed that learning was slow when the learning rate was very small, I had to increase the learning rate to allow the model to converge faster. Some studies have shown that there are links between batch size and learning rate, and when learning rate increases, it is suggested to increase the batch size as well, to simulate similar effects as decaying the learning rate (aka simulated annealing). Hence I also increased the batch size to 128.

These two were the main parameters tuned, there could be other parameters tuned such as dropout rate as well, but I observed that a dropout rate of 0.5 works well for this case, so I did not tune it much.

**What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?**

Convolutional networks have the property of ‘spatial invariance’, meaning they learn to recognise image features anywhere in your image. For example, it does not matter if an edge is detected on the left side or right side of a picture, as long as an edge is detected, its an important image feature to recognise the image. 

Dropout switches on and off the connections between filters/units with dropout in between in the entire network. This helps the network learn only the most important features in the images and prevents the network from overfitting to the training data. It increases the accuracy on the validation set and hence the testing set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web and these images are after resizing to fit the 32x32x3 for the model input:

![alt text][https://github.com/looboon/CarND-Traffic-Sign-Classifier-Project/blob/master/markup_images/new_web_images.png)

In general, we can see that there is a loss of image quality and detail after resizing and that makes the pictures harder to classify in general.

The first image should be quite simple to classify as its a nice picture of the turn right ahead sign. However, the training data does not have a lot of turn right ahead images in the dataset, so that might prove to be a challenge.

The second image is challenging, as not only is it an angled view, the loss of quality due to resizing makes it even tougher for humans to recognise the image, much less the CNN.

The third image also suffers from a loss of detail that makes this challenging, but important features in the sign are still kept, hence it is still possible for the model to classify this correctly.

The fourth and fifth images are straightforward, but because the fifth image is angled sideways, it may pose a slight difficultly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead  	| Turn right ahead   									| 
| Slippery road     			| Slippery road 										|
| Children crossing					| Children crossing						|
| Speed limit (30km/h)	  		| Speed limit (30km/h) 				|
| Stop			| Stop      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. While lower than the accuracy on the test set, I feel that the major reason for the lower accuracy than the actual testing accuracy is that the images downloaded from the internet were originally of higher overall quality (resolution) than the training and testing images. Resizing to 32x32x3 made quite abit of difference in terms of the lost quality, especially for the second sign, which even I myself was unable to recognise at first glance (granted that I did not have experience in road signs, but at first glance I could not tell at all it was a car slipping on the road.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a turn right ahead (probability of 0.985), and the image does contain a turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .985         			| Turn right ahead  									| 
| .010     				| Ahead only 										|
| .004					| Roundabout mandatory											|
| .001	      			| Turn left ahead					 				|
| .000				    | Keep left      							|


For the second image, the model was quite uncertain what image was it, with only a 0.361 probability of it being a dangerous curve to the right, but the image is actually slippery road. The probability of it being slippery road sign was 0.108. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .361         			| Dangerous curve to the right  									| 
| .189     				| Speed limit (120km/h) 										|
| .108					| Slippery road											|
| .106	      			| No entry					 				|
| .065				    | No passing for vehicles over 3.5 metric tons      							|

For the third image, the model is relatively sure that this is a children crossing (probability of 0.993) and it was correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .993         			| Children crossing  									| 
| .003     				| Road work 										|
| .002					| Right-of-way at the next intersection											|
| .000	      			| Pedestrians					 				|
| .000				    | Dangerous curve to the right      							|

For the fourth image, the model is very sure that this is a 30kmph speed limit sign (probability of 1.0) and it was correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000         			| Speed limit (30km/h)  									| 
| .000     				| Speed limit (20km/h) 										|
| .000					| Speed limit (80km/h)											|
| .000	      			| Speed limit (100km/h)					 				|
| .000				    | Speed limit (70km/h)      							|

For the fifth image, the model is very sure that this is a stop sign (probability of 1.0) and it was correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000         			| Stop  									| 
| .000     				| No entry 										|
| .000					| Yield											|
| .000	      			| Speed limit (60km/h)					 				|
| .000				    | No passing for vehicles over 3.5 metric tons      							|


