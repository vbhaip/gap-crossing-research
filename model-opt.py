# This file creates the optical flow model and trains it on the corresponding training dataset. It saves the weights of
# the model to a hdf5 file.

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Conv3D, Flatten, Dropout, MaxPooling2D, MaxPooling3D, Activation
from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import os
import csv
# from keras.applications.vgg16 import VGG16
# model = VGG16()
import cv2

from keras.callbacks import ModelCheckpoint
import pickle as pk


width = 100
height = 100
frames_input = 11

np.set_printoptions(precision=4, suppress=True)

counter = 0

# np.random.seed(1730)


X = []
y = [] 


for (dirpath, dirname, filenames) in os.walk("data-OPT-v1.1"):

	if len(filenames) > 1:

		for filename in filenames:
			# print(dirpath)
			#train or validation
			typ = dirpath.split("\\")[-2]

			#attempt or normal
			clas = dirpath.split("\\")[-1]

			if filename.endswith(".npy") and typ == "train":
				
				frame = np.load(dirpath + "\\" + filename)

				X.append(frame)




				if clas == "attempt":
					y.append([1,0])
				else:
					y.append([0,1])


				X.append(np.fliplr(frame))

				if clas == "attempt":
					y.append([1,0])
				else:
					y.append([0,1])


				counter += 2

X = np.asarray(X)

X = X /255.0

print(X.shape)

X.reshape(frames_input, width, height, counter)


X = np.expand_dims(X, axis=4)

print(X.shape)

y = np.asarray(y)



#this shuffles up the data, necessary bc when we do validation_split, it just picks consecutive data items
assert(len(X) == len(y))
p = np.random.permutation(len(X))
X, y = X[p], y[p]


data = [X, y]



# data = np.array(data)

# np.save("video-class-data-x.npy", X)
# np.save("video-class-data-y.npy", y)



model = Sequential()
model.add(Conv3D(8, (2,2,2), input_shape=(frames_input, width, height, 1)))
model.add(Activation('relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model.add(Conv3D(4, (4,4,4)))
model.add(Activation('relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(32))
# model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(16))
# model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))


# sgd = optimizers.SGD(lr=100)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.load_weights("weights-v1/weights-01-0.95.hdf5")


print(model.predict(X))
# X = X[1]
# print(X.shape)
# print(model.predict(np.expand_dims(X[1], axis=0)))



checkpoint = ModelCheckpoint("weights-opt-v1.1/weights-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(x= X, y = y, batch_size = 5, epochs = 10, validation_split = 0.2, callbacks=callbacks_list, shuffle=True)
