[//]: # (Image References)

[image_0]: ./markdown_source/background.jpg "Background"
[graph_1]: ./markdown_source/trainset_class_count.png "Graph 1"
[image_1]: ./markdown_source/raw.png "Raw 1"
[image_2]: ./markdown_source/gray.png "Gray 1"
[image_3]: ./markdown_source/lenet5.png "LeNet-5"
[image_4]: ./markdown_source/test_images.png "test"



## **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

    - Load the data set (see below for links to the project data set)
    - Explore, summarize and visualize the data set
    - Design, train and test a model architecture
    - Use the model to make predictions on new images
    - Analyze the softmax probabilities of the new images
    - Summarize the results with a written report

![alt text][image_0]

### **Project Rubric Self-Assessment:**
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. 

 
---
**Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.**

You're reading it! and here is a link to my [project code](https://github.com/lphucthinh40/cnn_traffic_sign_classifier/blob/master/Traffic-Sign-Detector.ipynb)

#### Data Set Summary & Exploration

**1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.**

I used the numpy library to calculate summary statistics of the traffic
signs dataset. Here is the result:


| Dataset    | Size   |
|------------|--------|
| train      | 34,799 |
| validation | 4,410  |
| test       | 12,630 |


Image shape: 32x32          
Number of unique classes: 43

**2. Include an exploratory visualization of the dataset.**

In my jupyter notebook, I displayed an example image for each unique class. I also calculated frequency of each class in the training set and plot a bar chart as below:

![alt text][graph_1]

This information can be useful in training and for further data preparation in the future. We can expect our model to perform better in predicting traffic sign classes that appear more often in the training set. 

#### Design and Test a Model Architecture

**1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.**

To preprocess image data, I used tensorflow.image functions including: 

    tf.image.rgb_to_grayscale() # to convert RGB image to Grayscale. 
    tf.image.per_image_standardization() # to normalize image with mean = 0 and standard deviation = 1

Here is an example of a traffic sign image before and after grayscaling.


![alt text][image_1]
![alt text][image_2]

To generate batches during training and evaluation, I created a dataloader using tensorflow's dataset object. The dataloader will shuffle the dataset, divide it into batches, apply preprocess function (via Dataset.map) for each image before feeding it into our model. Below is the code I used to create the dataloader:

    def dataloader(X, y):
        return tf.data.Dataset.from_tensor_slices((X, y))
                              .shuffle(len(X))         
                              .map(preprocess)          
                              .batch(BATCH_SIZE)        
                              .prefetch(1)

In the future, I will try to add data augmentation as a part of my data input pipeline.

**2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.**

My model architecture is based on the original LeNet-5. The only difference is that, since there are 43 unique classes, my final layer now has 43 output nodes. Below is a description of my model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 Grayscaled image						| 
| Convolution 5x5     	| 6 filters, 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5       | 16 filters, 1x1 stride, valid padding, outputs 10x10x16     |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Fully connected		| inputs: 400, outputs: 120 					|
| Fully connected       | inputs: 120, outputs: 84                      |
| Fully connected       | inputs: 84, outputs: 43                       |

![alt text][image_3]
LeNet-5 architecture as published in the original paper.

**3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.**

To train the model, I computed cross-entropy loss, performed Adam optimization and updated trained parameters for every batch of size 32, generated by my dataloader. Adam optimizer has default tensorflow settings with beta_1=0.9, beta_2=0.999, epsilon=1e-07. The learning rate was 0.001 - relatively low to avoid overshooting. I trained the model over 25 epochs. For every epoch, I also calculate the accuracy of my model on both train and validation set. 

**4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.**

After training the model through 25 epochs, I obtained the accuracy as follow:

| Dataset    | Accuracy   |
|------------|------------|
| train      | 0.99563    |
| validation | 0.94467    |
| test       | 0.91473    |

I believe there is still a lot that can be done to improve the overal accuracy of my model. One of the reason I did not have enough time to improve the current accuracy was because it took me quite some time to get familiar with Tensorflow 2.0 workflow. I was trying to avoid using Tensorflow-Keras API as well as not relying on tf.compat.v1 to reuse the tensorflow 1.0 code from the previous quiz. I just wanted to write a raw implementation in tensorlow 2.0, using only tensorflow 2.0 supported functions. In the future, I am going to try dropout and regularization along with data augmentation to improve the accuracy of my model.

I actually tried Tensorflow-Keras Sequential API before working on the raw implementation for this final submission. With the Keras approach, I added L2 regularization, and Dropout for the two dense layers to reduce chances of overfitting. After 83 epochs (it was actually 100 epochs with early stopping), I obtained the accuracy of 0.9558 for validation set, and 0.9386 for test set. It was quite surprising that my raw implementation reached 0.94 after just 25 epochs. Previously, I had a hard time reaching 0.94 with Sequential API without Dropout and Regularization. I also added my jupyter notebook with Sequential API approach in 'misc' folder for reference.  

#### Test a Model on New Images

**1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.**

Here are eight German traffic signs that I found on the web:

![alt text][image_4]

The 3rd, 7th, and 8th images could be difficult to classify because the traffic signs are not centered and there are also noises and other objects 
in these images.

**2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).**

Here are the results of the prediction:

| Id |                 Image                 |                Predict                |
|:--:|:-------------------------------------:|:-------------------------------------:|
| 1  | Road work                             | Road work                             |
| 2  | Right-of-way at the next intersection | Right-of-way at the next intersection |
| 3  | Stop                                  | No entry                              |
| 4  | Priority road                         | Priority road                         |
| 5  | No entry                              | No entry                              |
| 6  | General caution                       | No entry                              |
| 7  | Speed limit (30km/h)                  | Road work                             |
| 8  | Priority road                         | Priority road                         |

The model was able to correctly guess 5 of the 8 traffic signs, which gives an accuracy of 62.5%. This is much lower than the accuracy on test set. However, we should consider the fact that the traffic signs in many of these images are not well croped and centered.

**3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)**

The code for making predictions on my final model is located in the last cell of the Ipython notebook. I also printed top-5 probabilities for each image there. I will not discuss images where the model made the correct prediction with certainty above 99% (this includes image 1, 2, 4, 5, and 8).

For the 3rd image, the model made the wrong prediction ('No entry' with 59% certainty). Its second-ranked prediction was actually the correct one, which is 'Stop sign' with 30% certainty. This is not a surprise since after resizing the image to 32x32, it turns out that the round shape of the sign is no longer reserved and the text 'stop' becomes very hard to read. The top five soft max probabilities were:

| rank | class id |    class name    | probability |
|:----:|:--------:|:----------------:|-------------|
| 1    | 17       | No entry         | 0.58927     |
| 2    | 14       | Stop             | 0.30453     |
| 3    | 13       | Yield            | 0.06750     |
| 4    | 12       | Priority road    | 0.03529     |
| 5    | 33       | Turn right ahead | 0.00235     |

For the 6th image, the model made the wrong prediction - 'No entry' with 100% certainty - instead of 'General caution'. It appears that even though the traffic sign is recognizable after being resized to 32x32, the fact that there is another instruction sign on the image did interfere with my model's prediction. For the 7th image, the model made the wrong prediction - 'Road work' with 100% certainty - instead of 'Speed limit (30km/h)'. Again, noisy background has caused some problem with my model prediction. A red trafffic sign with green blackground may not preserve a very good contrast after being converted to grayscale.


