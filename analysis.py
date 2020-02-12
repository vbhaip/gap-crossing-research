# This model deals with the bulk of the angle/center of mass calculation for the model. It gets summaries for 
# videos on statistics regarding the amount of gap-crossing attempts and the lengths of each one.
# It also implements techniques to try to fit an ellipse on the fly.


import sys
import numpy as np
import csv
import cv2
import pickle as pickle
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Conv3D, Flatten, Dropout, MaxPooling2D, MaxPooling3D, Activation
from keras import optimizers
from keras.preprocessing.image import array_to_img, img_to_array, load_img
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

import argparse





parser = argparse.ArgumentParser()
parser.add_argument("--saveimgs", help="save figures to files", action="store_true")
parser.add_argument("--display", help="display any figures to the terminal (slows it down a lot)", action="store_true")
parser.add_argument("--savevid", help="saves vid w/ center of mass drawn over", action="store_true")
parser.add_argument("filename", type=str,
                    help="specify avi file location to access")

args = parser.parse_args()


#just some random seed to keep it consistent when testing
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



# model is the normal model that takes in the raw frames

model = Sequential()
model.add(Conv3D(8, (2,2,2), input_shape=(frames_input, width, height, 1)))
model.add(Activation('relu'))
model.add(Conv3D(4, (4,4,4)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.load_weights("weights/normal-0.84.hdf5")


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



#model2 is the optical flow model

model2 = Sequential()
model2.add(Conv3D(8, (2,2,2), input_shape=(frames_input, width, height, 1)))
model2.add(Activation('relu'))
model2.add(Conv3D(4, (4,4,4)))
model2.add(Activation('relu'))
model2.add(Flatten())
model2.add(Dense(32))
model2.add(Activation('relu'))
model2.add(Dense(16))
model2.add(Activation('relu'))
model2.add(Dense(2))
model2.add(Activation('softmax'))

model2.load_weights("weights/opt-0.87.hdf5")


model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


'''
Input: row from csv annotations file as a list,
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
			# print(avg_ang)

			if frames is not None:

				frames = np.expand_dims(frames, axis=3)
				frames = np.expand_dims(frames, axis=0)
				

				# normalizes the frames to 0 and 1 range
				frames = frames/255.0
				pred = model.predict(frames)
				pred = pred.tolist()[0]



				frames2 = np.expand_dims(frames2, axis=3)
				frames2 = np.expand_dims(frames2, axis=0)


				frames2 = frames2/255.0
				pred2 = model2.predict(frames2)
				pred2 = pred2.tolist()[0]

				print("Gap-crossing prediction at frame " + str(framenum) + ": " + str((pred2[0] + pred[0])/2))

				#if the average of the networks is above 50%, then we want to analyze this event
				if(abs(pred2[0] + pred[0])/2.0 > 0.5):
					print("Predicted gap-crossing event at frame " + str(framenum))

					# sets the video frame to 30 frames before the event location - this allows us to look at the
					# nearby area when a gap-crossing event occurs
					cap.set(1, framenum - 30)

					# name is just a unique identifier to locate vids - it can be changed
					name = vid_name.split("/")[-1].split(".")[0] + "-accuracy-" + str(abs(pred2[0] + pred[0])/2.0) + "-angle-" + str(avg_ang)
					


					if(args.savevid):
						new_vid = cv2.VideoWriter(name + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 15.0, (int(cap.get(3)), int(cap.get(4))))


					# correspond to the center of masses according to each dimension across time
					x_list = []
					y_list = []

					# keeps track of magnitudes of optical flow frames across time
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
							eigenEllipseFitting(mag, name)

						mx, my = getCenterMass(mag)

						x_list.append(mx)
						y_list.append(my)

					genSumPlot(mag_list, name)
					genHisto2D(mag_list, name)


					if(args.savevid):

						# creates a rolling average for the center of masses
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

							# draws red dot on frame
							for delx in range(-5,5):
								for dely in range(-5, 5):
									towrite[(int)(y_roll[x-framenum+30]) + dely, (int)(x_roll[x-framenum+30])+delx,...] = [0,0,255]

							# writes frame to video
							new_vid.write(towrite)

						new_vid.release()
					# saveVidFromNPY(frames, "somevid" + str(avg_ang) + ".avi")

	else:
		print(vid_name + " does not exist.\n")
		


# writes numpy frames with the given name to a file
def saveVidFromNPY(frames, name):
	new_vid = cv2.VideoWriter(name, -1, 1, (width, height))

	for x in range(0, 11):
		new_vid.write(frames[x])

	new_vid.release()
	print("Saved vid with angle")


# generates a small summary for the length of gap-crossing attempts and the number of attempts
def getSummary(preds):

	# min frame is 30, max frame is 565, interval of 5 frames
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


# saves the given video at the framenumber to a specific frame and returns that value
# this also calculates the naiive approach to getting the average angle, with a weighted average
def saveDataPoint(video, vid_name, framenum):

	avg_ang = 0.0

	tot_ang = 0.0

	temp_ang = 0.0


	# frames stores the normal frames and frames2 stores the optical flow frames
	frames = []
	frames2 = []
	
	length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


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
		
		# sums all the angles from the surrounding contextual frames and also stores the average angle as
		# the angle at the timestamp where the model actually predicted a gap-crossing event
		if(x == framenum):
			frame, avg_ang, _ = getOptFlowFrame(video)
		else:
			frame, temp_ang, _ = getOptFlowFrame(video)

			tot_ang += temp_ang

		frame = cleanFrame(frame)
		frames2.append(frame)

	frames = np.asarray(frames)
	frames2 = np.asarray(frames2)


	# this is a way to average all the angles across all the frames, it gives more weight to the angle at the
	# actual location of the frame where the gap-crossing event is located
	# the weighting for this was arbitrarily decided based off limited experimenting
	tot_ang = tot_ang + avg_ang*3
	tot_ang /= (FRAMECTX*2+3)
	avg_ang = tot_ang

	return (frames, frames2, avg_ang)



# calculates the weighted average for the angle by thresholding and then using the optical flow magnitude matrix
# as the weights
def getAngle(ang, mat):
	cut_off = np.percentile(mat, PERCENT_CUTOFF)

	mat[mat < cut_off] = 0

	return np.rad2deg(np.average(ang, weights=mat))



# returns the center of mass based off the magnitude matrix of the optical flow frame 
def getCenterMass(matrix):

	cut_off = np.percentile(matrix, PERCENT_CUTOFF)
	matrix[matrix < cut_off] = 0

	# sklearn's package to calculate the center of mass using a weighted technique
	my, mx = ndimage.measurements.center_of_mass(matrix)

	mx = int(round(mx))
	my = int(round(my))


	return mx, my


# calculates the covariance matrix of the magnitude matrix of the optical flow frame and calculates the eigenvectors
# and eigenvalues
# uses these eigenvectors to fit an ellipse and calculates the center of mass and theta based off the ellipse

def eigenEllipseFitting(matrix, name):


	# creates histogram of frame at a single point
	flat = matrix.flatten()
	flat = flat[flat > 0]
	histo = plt.hist(flat, bins=100)
	# plt.show()


	# gets coordinates for each relevant point for calculating the iegenvectors 
	datamod = np.argwhere(matrix > 0)

	# covariance matrix calculation
	sigma = np.cov(datamod, rowvar = False)


	# print(sigma)
	# print(len(sigma))

	# recalculates center of mass based off traditional method - needed because eigenvectors give shape of ellipse but
	# not location of the ellipse
	mu = getCenterMass(matrix)
	# print("mu" + str(mu))


	# calculates the eigenvectors (evals is eigenvals and evecs is eigenvecs)
	evals, evecs = la.eigh(sigma)

	# print(evals)
	# print(len(evals))
	# print(evecs)
	# print(len(evecs[0]))


	# gets theta value from eigenvectors
	x, y = evecs[:, 0]
	theta = np.degrees(np.arctan2(y, x))

	ax = plt.subplot(111)

	w, h = 2 * np.sqrt(evals)


	# plots optical flow matrix with ellipse overlayed
	ax.imshow(matrix)
	plt.colorbar(ax.imshow(matrix))
	ax.add_artist(Ellipse(mu, w, h, theta, fill=False))

	plt.title("Ellipse Fitting on Augmented Optical Flow")

	if(args.saveimgs):
		plt.savefig(name + "-ellipse-fit.png")

	if(args.display):
		plt.show()
		
	plt.clf()

	

# generates histogram with the optical flow vector magnitudes across all the frames in the time of an event
def genSumPlot(matrix, name):

	matrix = np.array(matrix)

	flat = matrix.flatten()

	histo = plt.hist(flat, bins=100, log=True)

	plt.title("Cumulative Sum Plot for Opt Flow Magnitudes Across Contextual Frames")

	plt.xlabel("Optical Flow Magnitude")

	plt.ylabel("Number of occurrences")


	if(args.saveimgs):
		plt.savefig(name + "-sumplot.png")

	if(args.display):
		plt.show()

	plt.clf()



# generates 2d histogram with the optical flow vector magnitudes to show movement
def genHisto2D(matrix, name):


	arr = []
	bin_edges = 100

	established_bin = False

	# creates bounds for image
	low = np.amin(matrix)
	high = np.amax(matrix)

	for x in matrix:
		flat = x.flatten()

		histo = np.histogram(flat, bins=100, range=(low, high))

		# this can be used for seeing the xth percentile visualized on the 2d histogram
		# val = (np.percentile(flat, 99) - low) // ((high-low)/100.0)
		# histo[0][int(val)] = 100000000000000

		arr.append(histo[0])


	arr = np.array(arr)

	histo_2d = arr.reshape(-1,100)

	my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))

	# when using a log scale, the 0s in a matrix give an error so this just sets the value of those elements to 
	# the smallest value on the corresponding color scale
	my_cmap.set_bad(my_cmap.colors[0])



	plt.imshow(histo_2d, cmap=my_cmap, norm=LogNorm())

	plt.colorbar()

	plt.title("2D Histogram of Optical Flow Vectors")
	plt.xlabel("Optical Flow Magnitude")
	plt.ylabel("Frame Number (middle is location of predicted event)")

	if(args.saveimgs):
		plt.savefig(name + "-2Dhisto.png")

	if(args.display):
		plt.show()

	plt.clf()


# takes in the video reader and returns the rgb representation of the optical flow, the angle matrix, and the 
# magnitude matrix
def getOptFlowFrame(cap):

	ret, frame1 = cap.read()
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255

	ret, frame2 = cap.read()
	next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


	# calculates optical flow - taken from opencv tutorial
	flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)


	# turns into polar form for magnitude and direction
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2

	# kernel is used for dilation and erosion (2,2) can be modified to adjust kernel size
	kernel = np.ones((2,2), np.uint8)
	bin_mag = mag.copy()
	
	# bin_mag is the binary mask for the optical flow magnitude matrix
	# thresholded bc the rest of the image should eventually 
	bin_mag[bin_mag <=  np.percentile(mag, 99)] = 0
	bin_mag[bin_mag >  np.percentile(mag, 99)] = 1

	# print("Non-zero: " + str(np.count_nonzero(bin_mag)))
	# print("Size: " + str(np.size(bin_mag)))


	# iterative erosion, iterations can be changed to suit situation
	bin_mag = cv2.erode(bin_mag,kernel,iterations = 2)

	# this does an erosion and then a dilation with the kernel
	opening = cv2.morphologyEx(bin_mag, cv2.MORPH_OPEN, kernel)
	mag = np.multiply(mag, opening)


	# this just shows that the erosion and dilation is actually working and reducing the optical flow vectors we're
	# dealing with
	# print("MAG Non-zero: " + str(np.count_nonzero(mag)))


	# calculates average angle and center of mass to return
	avg_ang = getAngle(ang, mag)
	mx, my = getCenterMass(mag)

	# visual matrix of the optical flow matrix 
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	
	return rgb, avg_ang, mag

'''
Input: frame as numpy array from opencv

Return: cleaned frame as numpy array - grayscale and reduced to 100x100
'''
def cleanFrame(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(frame, (100,100))
	return frame





finished_vids = []

#for actual analysis, you can use this to keep track of videos already completed

# with open("analysis.txt", "r") as f_read:

# 	for line in f_read.readlines():
# 		name = line.split(" ")[0]
# 		finished_vids.append(name)



# can use this to make the system look at a specific video rather than going through all of it
# helpful for A/B testing to compare techniques
# need_to_check_vid = "GMR_SS00010__UAS_Shi_ts1_3_0001_20130724T123424(5.0_01)_0.avi"


# traverses videos to analyze

# for (dirpath, dirname, filenames) in os.walk("analysis-vids"):
# 	print(dirpath)
# 	count = 0
# 	for filename in filenames:
# 		print(filename)

# 		# can add this as a condition "and filename not in need_to_check_vids[0]" if you want to look at a specific vid
# 		if(filename[-5:] == "0.avi" and filename not in finished_vids and dirpath.split("\\")[-1] == "attempt"):
# 			# x = time.time()
# 			readVid(dirpath + "\\" + filename)
# 			# print(time.time() - x)
# 			count += 1


# 		else:
# 			"Already finished this vid"



readVid(args.filename)



