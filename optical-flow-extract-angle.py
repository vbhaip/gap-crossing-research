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

from scipy import ndimage

from numpy import linalg as la

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import matplotlib


from matplotlib.colors import LogNorm

import copy


# np.set_printoptions(threshold=np.inf)

np.random.seed(1730)


#means we want to skip x frames for every frame we want to use
FRAMEGAP = 5

#means x frames before and after are included in the data point
FRAMECTX = 5


FRAME_MEASURE_LENGTH = 30

IMG_WIDTH = 1024
IMG_HEIGHT = 512

width = 100
height = 100
frames_input = 11

PERCENT_CUTOFF = 75

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

model2.load_weights("weights/opt-0.87.hdf5")


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
	avg_ang = 0.0

	if(os.path.exists(vid_name)):
		print(vid_name + " exists.\n")
		cap = cv2.VideoCapture(vid_name)
		# print(row)

		final_preds = []

		for framenum in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-FRAMEGAP*FRAMECTX-5, FRAME_MEASURE_LENGTH):
			frames, frames2, avg_ang = saveDataPoint(cap, vid_name, framenum)
			print(avg_ang)

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

				print(str((pred2[0] + pred[0])/2))
				if(abs(pred2[0] + pred[0])/2.0 > 0.5):
					print("event")
					print(str(framenum - 30))
					cap.set(1, framenum - 30)

					name = "v2-rollavgcm-pc" + str(PERCENT_CUTOFF) + "-anglevid" + str((pred[0])) + "opt" + str((pred2[0])) + "--" + str(avg_ang)
					new_vid = cv2.VideoWriter(name + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 15.0, (int(cap.get(3)), int(cap.get(4))))

					x_list = []
					y_list = []

					mag_list = []
					for x in range(framenum - 30, framenum + 30):
						cap.set(1, x)
						ret, towrite = cap.read()
						# cv2.imshow('frame', towrite)
						# if cv2.waitKey(500) & 0xFF == ord('q'):
						# 	break
						cap.set(1,x)
						temp_frame, _, mag = getOptFlowFrame(cap)
						mag_list.append(mag)
						if(x == framenum):
							eigenStuff(mag, name)

						mx, my = getCenterMass(mag)

						x_list.append(mx)
						y_list.append(my)

					genSumPlot(mag_list, name)
					genHisto2D(mag_list, name)
					x_roll = np.convolve(x_list, np.ones((5,))/5, mode='valid')
					y_roll = np.convolve(y_list, np.ones((5,))/5, mode='valid')
					count = 0
					for x in range(framenum - 30, framenum + 30):

						if(x - framenum + 30 >= len(x_roll)):
							break

						cap.set(1, x)
						ret, towrite = cap.read()
						# cv2.imshow('frame', towrite)
						# if cv2.waitKey(500) & 0xFF == ord('q'):
						# 	break
						cap.set(1,x)
						for delx in range(-5,5):
							for dely in range(-5, 5):
								towrite[(int)(y_roll[x-framenum+30]) + dely, (int)(x_roll[x-framenum+30])+delx,...] = [0,0,255]


						new_vid.write(towrite)

						# cv2.imshow('frame', towrite)
						# if cv2.waitKey(1) & 0xFF == ord('q'):
						# 	break

					new_vid.release()
					# saveVidFromNPY(frames, "somevid" + str(avg_ang) + ".avi")
				
		# 		# print(pred)
		# 		# print("Frame " + str(framenum) + " pred: " + str(pred.index(max(pred))) + " " + str(max(pred)) + "\n")
		# 		# print(str(framenum) + " " + str(pred[0]) + " " + str(pred2[0]))
		# 		final_preds.append((pred[0]+pred2[0])/2)

		# # rolling avg w frame width of 5
		# roll_avg = np.convolve(final_preds, np.ones((5,))/5, mode='valid')
		# summary = getSummary(roll_avg)

		# with open("analysis.txt", "a") as f:
		# 	f.write(str(vid_name.split("\\")[-1]))
		# 	f.write(" ")
		# 	f.write(str(summary))
		# 	f.write(" ")
		# 	f.write(str(roll_avg))
		# 	f.write(" ")
		# 	f.write("\n")

	else:
		print(vid_name + " does not exist.\n")
		

	# print(row)
	# print("\n\n")


def saveVidFromNPY(frames, name):
	new_vid = cv2.VideoWriter(name, -1, 1, (width, height))


	for x in range(0, 11):
		new_vid.write(frames[x])

	new_vid.release()
	print("Saved vid w angle!")



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

	avg_ang = 0.0

	tot_ang = 0.0

	temp_ang = 0.0

	frames = []
	frames2 = []
	
	length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


	# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

	if(framenum-FRAMECTX*(FRAMEGAP+1) < 0 or 
		framenum+FRAMECTX*(FRAMEGAP+1) > length):
			print("Out of range, skipping sample. \n")
			return (None,None,None)
	for x in range(framenum - FRAMECTX*(FRAMEGAP+1), framenum + FRAMECTX*(FRAMEGAP+1) + 1, FRAMEGAP+1):
		video.set(1, x)
		ret, frameraw = video.read()

		# frameraw = fgbg.apply(frameraw)

		frame = cleanFrame(frameraw)
		frames.append(frame)

		video.set(1, x)
		
		if(x == framenum):
			frame, avg_ang, _ = getOptFlowFrame(video)
		else:
			frame, temp_ang, _ = getOptFlowFrame(video)

			tot_ang += temp_ang

		frame = cleanFrame(frame)
		frames2.append(frame)

	frames = np.asarray(frames)
	frames2 = np.asarray(frames2)


	#################

	# bckg = frames.mean()


	tot_ang = tot_ang + avg_ang*3

	tot_ang /= (FRAMECTX*2+3)

	avg_ang = tot_ang

	return (frames, frames2, avg_ang)


def getAngle(ang, mat):
	cut_off = np.percentile(mat, PERCENT_CUTOFF)

	mat[mat < cut_off] = 0

	# thresholded = np.multiply(ang, mat)

	return np.rad2deg(np.average(ang, weights=mat))




def getCenterMass(matrix):
	# new_matrix = np.empty_like(matrix)

	cut_off = np.percentile(matrix, PERCENT_CUTOFF)
	matrix[matrix < cut_off] = 0
	# print(matrix)
	# print(matrix.shape)
	my, mx = ndimage.measurements.center_of_mass(matrix)

	# mx /= IMG_WIDTH		
	# my /= IMG_HEIGHT
	# my = 1 - my

	# my = IMG_HEIGHT - my


	mx = int(round(mx))
	my = int(round(my))
	# print(mx)
	# print(my)
	return mx, my


def eigenStuff(matrix, name):

	flat = matrix.flatten()

	flat = flat[flat > 0]

	histo = plt.hist(flat[flat > np.percentile(flat, 0)], bins=100)
	plt.show()


	# matrix[matrix < np.percentile(matrix, PERCENT_CUTOFF)] = 0

	# print(matrix)

	datamod = np.argwhere(matrix > 0)

	# datamod_weights = matrix[matrix > 0]

	# print(datamod)


	sigma = np.cov(datamod, rowvar = False)


	# sigma = np.cov(matrix)

	print(sigma)
	print(len(sigma))
	# mu = datamod.mean(axis = 0)
	mu = getCenterMass(matrix)
	print("mu" + str(mu))

	evals, evecs = la.eigh(sigma)

	print(evals)
	print(len(evals))
	print(evecs)
	print(len(evecs[0]))

	x, y = evecs[:, 0]
	theta = np.degrees(np.arctan2(y, x))

	ax = plt.subplot(111)

	w, h = 2 * np.sqrt(evals)


	ax.imshow(matrix)

	plt.colorbar(ax.imshow(matrix))

	ax.add_artist(Ellipse(mu, w, h, theta, fill=False))

	plt.savefig(str(PERCENT_CUTOFF) + name + "non-weight.png")

	plt.show()

	


def genSumPlot(matrix, name):


	# for x in matrix:
	matrix = np.array(matrix)

	flat = matrix.flatten()

	# flat = flat[flat > 0]

	histo = plt.hist(flat, bins=100, log=True)
	plt.show()


	# datamod_weights = matrix[matrix > 0]

	# print(datamod)

	# print(evals)


def genHisto2D(matrix, name):


	arr = []
	bin_edges = 100

	established_bin = False


	low = np.amin(matrix)
	high = np.amax(matrix)

	for x in matrix:
		flat = x.flatten()
		# print(flat)

		histo = np.histogram(flat, bins=100, range=(low, high))
		# print(len(histo[0]))
		# print(histo)
		# print("\n\n")

		if not established_bin:
			bin_edges = histo[1]
			established_bin = True

		# val = (np.percentile(flat, 99) - low) // ((high-low)/100.0)

		# histo[0][int(val)] = 100000000000000

		arr.append(histo[0])


	arr = np.array(arr)

	histo_2d = arr.reshape(-1,100)

	my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
	my_cmap.set_bad(my_cmap.colors[0])



	plt.imshow(histo_2d, cmap=my_cmap, norm=LogNorm())

	plt.colorbar()
	plt.show()



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


	kernel = np.ones((2,2), np.uint8)
	bin_mag = mag.copy()
	
	bin_mag[bin_mag <=  np.percentile(mag, 99)] = 0
	bin_mag[bin_mag >  np.percentile(mag, 99)] = 1

	print("Non-zero: " + str(np.count_nonzero(bin_mag)))
	print("Size: " + str(np.size(bin_mag)))



	bin_mag = cv2.erode(bin_mag,kernel,iterations = 2)


	opening = cv2.morphologyEx(bin_mag, cv2.MORPH_OPEN, kernel)
	mag = np.multiply(mag, opening)

	print("MAG Non-zero: " + str(np.count_nonzero(mag)))

	avg_ang = getAngle(ang, mag)
	# cut_off = np.percentile(ang, PERCENT_CUTOFF)

	# ang[ang < cut_off] = 0

	# avg_ang = np.rad2deg(np.average(ang, weights=mag))

	mx, my = getCenterMass(mag)

	# mag[mag > 0.005] = 0

	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	# print(rgb.shape)

	

	return rgb, avg_ang, mag

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

with open("analysis.txt", "r") as f_read:

	for line in f_read.readlines():
		name = line.split(" ")[0]
		# print(name)
		finished_vids.append(name)

# print(finished_vids)


################################
#This just means that we're looking through all vids each time
finished_vids = []
# ..\\..\\Downloads


need_to_check_vids = ["GMR_SS00010__UAS_Shi_ts1_3_0001_20130724T123424(5.0_01)_0.avi"]

ind = 0
# types = ["GMR_SS00011__UAS_DL_eGFPKir21_3_0066", "GMR_SS00011__UAS_Shi_ts1_3_0001", "GMR_SS00013__UAS_DL_Shi_ts1_3_0033", "GMR_SS00014__UAS_DL_eGFPKir21_3_0066", "GMR_SS00015__UAS_DL_eGFPKir21_3_0066"
# "GMR_SS00016__UAS_DL_Shi_ts1_3_0033", "GMR_SS00017__UAS_DL_Shi_ts1_3_0033", "GMR_SS00018__UAS_DL_Shi_ts1_3_0033", 
# "GMR_SS00019__UAS_CSMH_Shi_ts1_3_0047", "GMR_SS00019__UAS_DL_eGFPKir21_3_0066", "GMR_SS00020__UAS_DL_eGFPKir21_3_0066", 
# "GMR_SS00020__UAS_DL_Shi_ts1_3_0033"]

# types = ["GMR_SS00019__UAS_DL_eGFPKir21_3_0066"]
for (dirpath, dirname, filenames) in os.walk("analysis-vids"):
	print(dirpath)
	count = 0
	for filename in filenames:
		print(filename)
		if(filename[-5:] == "0.avi" and filename not in finished_vids and dirpath.split("\\")[-1] == "attempt" and filename not in need_to_check_vids[0]):
			x = time.time()
			readVid(dirpath + "\\" + filename)
			print(time.time() - x)
			count += 1
		# if(count > 30):
		# 	count = 0
		# 	break

		else:
			"Already finished this vid"

