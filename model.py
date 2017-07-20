import csv
import cv2
import numpy as np

# Saving the path for each of the data sets
paths = ['data','data_opp','data_smooth', 'data_rec']

images = []
measurements = []

# For each of the data sets read the csv files with the driving_logs,
# and all three of the camera images
for path in paths:
	with open(path + '/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			for i in range(3):
                		source_path = line[i]
                		filename = source_path.split('/')[-1]
                		current_path = path + '/IMG/' + filename
                		image = cv2.imread(current_path)
                		images.append(image)
			correction = 0.2
			measurement = float(line[3])
			measurements.append(measurement)
			measurements.append(measurement+correction)
			measurements.append(measurement-correction)

# Augment the data by fliping the images,
# and multiplying the steering angles by -1 (make left turns, right turns)
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

# Create the training data arrays
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Import Keras models and layers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Model developed based on the end-to-end NVidia neural network
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# Train network
model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

# Save trained network
model.save('model.h5')
