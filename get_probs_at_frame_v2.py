import numpy as np
import csv
import cv2
import pickle as pickle
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Conv3D, Flatten, Dropout, MaxPooling2D, MaxPooling3D, Activation
from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
# from keras.applications.vgg16 import VGG16
# model = VGG16()
import cv2

from keras.callbacks import ModelCheckpoint
import time


np.random.seed(1730)


#means we want to skip x frames for every frame we want to use
FRAMEGAP = 5

#means x frames before and after are included in the data point
FRAMECTX = 5


FRAME_MEASURE_LENGTH = 15

width = 100
height = 100
frames_input = 11

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

model.load_weights("weights-normal-v1/weights-03-0.84.hdf5")


# sgd = optimizers.SGD(lr=100)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model2 = Sequential()
model2.add(Conv3D(8, (2,2,2), input_shape=(frames_input, width, height, 1)))
model2.add(Activation('relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model2.add(Conv3D(4, (4,4,4)))
model2.add(Activation('relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Activation('relu'))
model2.add(Flatten())
model2.add(Dense(32))
model2.add(Activation('relu'))
model2.add(Dense(16))
model2.add(Activation('relu'))
model2.add(Dense(2))
model2.add(Activation('softmax'))

model2.load_weights("weights-opt-v1.1/weights-03-0.87.hdf5")


# sgd = optimizers.SGD(lr=100)

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# f = open("analysis.txt", "w")

'''
Input: row from csv file as a list,
first item is name of video, every pair following is 
label then frame number

Process: passes each event and frame number into generateFrame
'''
def readVid(vid_name):

	if(os.path.exists(vid_name)):
		print(vid_name + " exists.\n")
		cap = cv2.VideoCapture(vid_name)
		# print(row)

		final_preds = []

		for framenum in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-FRAMEGAP*FRAMECTX-5, FRAME_MEASURE_LENGTH):
			frames, frames2 = saveDataPoint(cap, vid_name, framenum)

			if frames is not None:
				# print(frames.shape)

				frames = np.expand_dims(frames, axis=3)
				frames = np.expand_dims(frames, axis=0)
				# print(frames.shape)
				# print(frames)
				# np.save("probs-test.npy", frames)

				frames = frames/255.0
				pred = model.predict(frames)
				pred = pred.tolist()[0]



				frames2 = np.expand_dims(frames2, axis=3)
				frames2 = np.expand_dims(frames2, axis=0)
				# print(frames.shape)
				# print(frames)
				# np.save("probs-test.npy", frames)

				frames2 = frames2/255.0
				pred2 = model2.predict(frames2)
				pred2 = pred2.tolist()[0]


				
				# print(pred)
				# print("Frame " + str(framenum) + " pred: " + str(pred.index(max(pred))) + " " + str(max(pred)) + "\n")
				# print(str(framenum) + " " + str(pred[0]) + " " + str(pred2[0]))
				final_preds.append((pred[0]+pred2[0])/2)

		# rolling avg w frame width of 5
		roll_avg = np.convolve(final_preds, np.ones((5,))/5, mode='valid')
		summary = getSummary(roll_avg)

		with open("analysis-dec-16-ss90-96-98.txt", "a") as f:
			f.write(str(vid_name.split("\\")[-1]))
			f.write(" ")
			f.write(str(summary))
			f.write(" ")
			f.write(str(roll_avg))
			f.write(" ")
			f.write("\n")

	else:
		print(vid_name + " does not exist.\n")
		

	# print(row)
	# print("\n\n")

def getSummary(preds):

	#min frame is 30, max frame is 565, interval of 5 frames
	summary = []
	in_attempt = False
	curr_length = 0
	start = -1

	for x in range(0, len(preds)):

		if(in_attempt):
			if(preds[x] > 0.5):
				curr_length += FRAME_MEASURE_LENGTH
			else:	
				summary.append((start, curr_length))

				start = -1
				curr_length = 0
				in_attempt = False

		else:
			if(preds[x] > 0.5):
				in_attempt = True
				start = FRAME_MEASURE_LENGTH*x+30

	return summary


def saveDataPoint(video, vid_name, framenum):


	frames = []
	frames2 = []
	
	length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

	if(framenum-FRAMECTX*(FRAMEGAP+1) < 0 or 
		framenum+FRAMECTX*(FRAMEGAP+1) > length):
			print("Out of range, skipping sample. \n")
			return (None,None)
	for x in range(framenum - FRAMECTX*(FRAMEGAP+1), framenum + FRAMECTX*(FRAMEGAP+1) + 1, FRAMEGAP+1):
		video.set(1, x)
		ret, frameraw = video.read()
		frame = cleanFrame(frameraw)
		frames.append(frame)

		video.set(1, x)
		frame = getOptFlowFrame(video)
		frame = cleanFrame(frame)
		frames2.append(frame)

	frames = np.asarray(frames)
	frames2 = np.asarray(frames2)

	return (frames, frames2)




def getOptFlowFrame(cap):

	ret, frame1 = cap.read()
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255
	# video = cv2.VideoWriter('temp_opt_flow.avi', cv2.VideoWriter_fourcc(*'XVID'),30,(1024,512))


	ret, frame2 = cap.read()
	next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

	flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2


	# mag[mag > 0.005] = 0

	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	return rgb

'''
Input: frame as numpy array from opencv

Return: cleaned frame as numpy array
'''
def cleanFrame(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(frame, (100,100))
	return frame


# readVid("GMR_SS00010__UAS_DL_eGFPKir21_3_0066_20130612T133023(3.5_01)_0.avi")


finished_vids = []

with open("analysis-dec-16-ss90-96-98.txt", "r") as f_read:

	for line in f_read.readlines():
		name = line.split(" ")[0]
		print(name)
		finished_vids.append(name)

print(finished_vids)
# ..\\..\\Downloads

ind = 0
# types = ["GMR_SS00011__UAS_DL_eGFPKir21_3_0066", "GMR_SS00011__UAS_Shi_ts1_3_0001", "GMR_SS00013__UAS_DL_Shi_ts1_3_0033", "GMR_SS00014__UAS_DL_eGFPKir21_3_0066", "GMR_SS00015__UAS_DL_eGFPKir21_3_0066"
# "GMR_SS00016__UAS_DL_Shi_ts1_3_0033", "GMR_SS00017__UAS_DL_Shi_ts1_3_0033", "GMR_SS00018__UAS_DL_Shi_ts1_3_0033", 
# "GMR_SS00019__UAS_CSMH_Shi_ts1_3_0047", "GMR_SS00019__UAS_DL_eGFPKir21_3_0066", "GMR_SS00020__UAS_DL_eGFPKir21_3_0066", 
# "GMR_SS00020__UAS_DL_Shi_ts1_3_0033"]

types = ["GMR_SS00090__UAS_TNT_2_0003", "GMR_SS00096__UAS_TNT_2_0003", "GMR_SS00098__UAS_TNT_2_0003"]

# types = ["GMR_SS00096__UAS_TNT_2_0003", "GMR_SS00098__UAS_TNT_2_0003"]

for (dirpath, dirname, filenames) in os.walk("data_new_ss90_96_98"):
	print(dirpath)
	count = 0
	for filename in filenames:
		if(filename[-5:] == "0.avi" and filename not in finished_vids and dirpath.split("\\")[-1] == "attempt" and dirpath.split("\\")[-2] in types):
			x = time.time()
			readVid(dirpath + "\\" + filename)
			print(time.time() - x)
			count += 1
		# if(count > 30):
		# 	count = 0
		# 	break

		else:
			"Already finished this vid"

