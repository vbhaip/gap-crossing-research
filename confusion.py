# This file just generates a confusion matrix for the normal model. Just helpful to see how the model performs across 
# different stats for false positive and false negatives


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Conv3D, Flatten, Dropout, MaxPooling2D, MaxPooling3D, Activation
from keras import optimizers
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import os
import csv
import cv2

from keras.callbacks import ModelCheckpoint
import pickle as pk

from sklearn.metrics import confusion_matrix

width = 100
height = 100
frames_input = 11


np.set_printoptions(precision=4, suppress=True)

# np.random.seed(1730)

counter = 0



X = []
y = [] 


for (dirpath, dirname, filenames) in os.walk("data-normal-v1"):

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

X.reshape(counter, width, height, frames_input, 1)


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
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.load_weights("weights/normal-0.84.hdf5")


# sgd = optimizers.SGD(lr=100)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




# checkpoint = ModelCheckpoint("weights-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]


y_hat = model.predict(X)

y_hat = np.argmax(y_hat, axis=1)

y = np.argmax(y, axis = 1)

stats = confusion_matrix(y, y_hat)

print(stats)

print(stats.ravel())

error = model.evaluate(x= X, y = y, batch_size=5)
print(error)
