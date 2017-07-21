# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[image1]: ./writeup_images/model.png "Model Visualization"
[image2]: ./writeup_images/original.jpg "Normal Image"
[image3]: ./writeup_images/rec1.jpg "Recovery Image"
[image4]: ./writeup_images/rec2.jpg "Recovery Image"
[image5]: ./writeup_images/rec3.jpg "Recovery Image"
[image6]: ./writeup_images/flipped.jpg "Flipped Image"
## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 a video recording of the vehicle driving autonomously around the track for one full lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of five convolutional neural networks: the first three with 5x5 filter sizes and depths between 24 and 48, and the last two with 3x3 filter sizes and depths of 64 both (model.py lines 53-57).

The model includes RELU layers to introduce nonlinearity (model.py line 59), and the data is normalized in the model using a Keras Lambda layer (model.py line 51). Also, the dataset were pre-processed by using a Keras Cropping2D layer to mask the parts of the images that might confuse the model. (model.py line 52)

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer with a keep probability of 0.5 to avoid overfitting (model.py lines 58). 

Another way to prevent the model from overfitting was to create different data sets (model.py line 6). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 66).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of four different collectin data methods. The main data was collected by driving the car around the track in both direction while trying to stay centered. Extra data was also collected focusing on making very smooth curves (using the mouse to turn the car) and getting the car back to the center, recovering from a mistake that could make the car go off track.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a convolution neural network similar to the NVidia End-to-End Deep Learning Architecture and from it test on autonomous mode and make small changes. I thought this model could be appropriate since it was developed for Self-Driving Cars.

I found that the first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I tried running on fewer epochs but the model didn't train so well.

So I added a few epochs back and to combat the overfitting, I decided to augment the data by collecting driving the car on training mode using different approaches like recovering and driving in the opposite direction. I also flipped the images using the cv2.flip function while multiplying the steering angles by -1.

I ran the simulator once again to see how well the car was driving. There were a few spots where the vehicle fell off the track and it even crashed on the side of the bridge. So I did the recovery lap again, being sure the car was positioned properly before driving back to the center of the track. I also added a Dropout layer with a keep probability of 0.5 and ran the simulator again.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. It did touch the yellow line a couple of times, but was able to recover and get back to the track.

#### 2. Final Model Architecture

The final model architecture (model.py lines 50-63) consisted of a convolution neural network with the following Keras layers and layer sizes: 
* Convolution2D layer with a 5x5 filter size and 24 features deep
* Convolution2D layer with a 5x5 filter size and 36 features deep
* Convolution2D layer with a 5x5 filter size and 48 features deep
* Convolution2D layer with a 3x3 filter size and 64 features deep
* Convolution2D layer with a 3x3 filter size and 64 features deep
* Dropout layer with keep probabilty of 0.5
* Activation RELu layer
* Flatten layer
* Fully-connected with 120 neurons
* Fully-connected with 84 neurons
* Fully-connected with 1 neuron (Steering angle)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving (both directions). Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover by itself in case it was driving out of the track. These images show what a recovery looks like starting from the right side:

![alt text][image3]|
-------------------|
![alt text][image4]|
-------------------|
![alt text][image5]|

I then repeated this process going in the other direction in order to get more data points. I also did two laps focusing on driving very smoothly around the curves.

To augment the data set, I also flipped images and angles. This allowed me to double the data with no effort, which helped me combat overfiting. For example, here is the center lane driving image that I displayed before when flipped:

![alt text][image6]

After the collection process, I had 24384 data points. I then preprocessed this data by normalizing to a range of [-0.5,0.5] and by cropping 70 pixels from the top of the image and 25 pixels from the bottom, without changing the sides. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs chosen was 6. I used an adam optimizer so that manually training the learning rate wasn't necessary.
