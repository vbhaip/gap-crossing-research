import numpy as np
import csv
import cv2
import pickle as pickle
import os

#means we want to skip x frames for every frame we want to use
FRAMEGAP = 5

#means x frames before and after are included in the data point
FRAMECTX = 5

'''
Input: row from csv file as a list,
first item is name of video, every pair following is 
label then frame number

Process: passes each event and frame number into generateFrame
'''
def readCSVRow(row):

	vid_name = row[0]

	if(os.path.exists("../vids/" + vid_name)):
		print(vid_name + " exists.\n")
		cap = cv2.VideoCapture("../vids/" + vid_name)
		# print(row)

		##2 so that it skips the actual label and just does the frame num
		event_frames = []

		for x in range(1, len(row) - 1, 2):
			label = row[x]
			framenum = row[x+1]


			if (label != "" and label != "type" and len(framenum) > 0):
				for y in range(-5, 6, 1):
					event_frames.append(int(framenum.strip())+y)

				saveDataPoint(cap, vid_name, label, int(framenum))

		# for x in range(1, 600, 100):
		# 	label = "nothing"
		# 	framenum = x

		# 	if(framenum not in event_frames):
		# 		saveDataPoint(cap, vid_name, label, int(framenum))



	else:
		print(vid_name + " does not exist.\n")
		

	# print(row)
	# print("\n\n")



def readCSVRowSideView(row):

	vid_name = row[0][:-5] + "1.avi"


	if(os.path.exists("../vidsp2/" + vid_name)):
		print(vid_name + " exists.\n")
		cap = cv2.VideoCapture("../vidsp2/" + vid_name)
		# print(row)

		##2 so that it skips the actual label and just does the frame num
		event_frames = []

		for x in range(1, len(row) - 1, 2):
			label = row[x]
			framenum = row[x+1]


			if (label != "" and label != "type" and len(framenum) > 0):
				for y in range(-5, 6, 1):
					event_frames.append(int(framenum.strip())+y)

				saveDataPoint(cap, vid_name, label, int(framenum))

		for x in range(1, 600, 100):
			label = "nothing"
			framenum = x

			if(framenum not in event_frames):
				saveDataPoint(cap, vid_name, label, int(framenum))



	else:
		print(vid_name + " does not exist.\n")
		

	# print(row)
	# print("\n\n")


def readCSVRowOptFlow(row):

	vid_name = row[0]

	if(os.path.exists("../vidsp2/" + vid_name)):
		print(vid_name + " exists.\n")
		cap = cv2.VideoCapture("../vidsp2/" + vid_name)
		# print(row)

		##2 so that it skips the actual label and just does the frame num
		event_frames = []

		for x in range(1, len(row) - 1, 2):
			label = row[x]
			framenum = row[x+1]


			if (label != "" and label != "type" and len(framenum) > 0):
				for y in range(-5, 6, 1):
					event_frames.append(int(framenum.strip())+y)

				saveDataPointOptFlow(cap, vid_name, label, int(framenum))

		for x in range(1, 600, 100):
			label = "nothing"
			framenum = x

			if(framenum not in event_frames):
				saveDataPointOptFlow(cap, vid_name, label, int(framenum))



	else:
		print(vid_name + " does not exist.\n")
		

	# print(row)
	# print("\n\n")



def readCSVRowSideViewOptFlow(row):

	vid_name = row[0][:-5] + "1.avi"


	if(os.path.exists("../vidsp2/" + vid_name)):
		print(vid_name + " exists.\n")
		cap = getOptFlowVid("../vidsp2/" + vid_name)
		# print(row)

		##2 so that it skips the actual label and just does the frame num
		event_frames = []

		for x in range(1, len(row) - 1, 2):
			label = row[x]
			framenum = row[x+1]


			if (label != "" and label != "type" and len(framenum) > 0):
				for y in range(-5, 6, 1):
					event_frames.append(int(framenum.strip())+y)

				# saveDataPoint(cap, "OptFlow" + vid_name, label, int(framenum))

		for x in range(1, 600, 100):
			label = "nothing"
			framenum = x

			# if(framenum not in event_frames):
				# saveDataPoint(cap, "OptFlow" + vid_name, label, int(framenum))



	else:
		print(vid_name + " does not exist.\n")
		

	# print(row)
	# print("\n\n")



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


def saveDataPointOptFlow(video, vid_name, label, framenum):

	if(not os.path.exists(label + "allvid-OPT/" + vid_name + str(framenum) + ".npy")):

		frames = []
		
		length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

		if(framenum-FRAMECTX*(FRAMEGAP+1) < 0 or 
			framenum+FRAMECTX*(FRAMEGAP+1) > length):
			print("Out of range, skipping sample. \n")
		else:
			for x in range(framenum - FRAMECTX*(FRAMEGAP+1),
				framenum + FRAMECTX*(FRAMEGAP+1) + 1, FRAMEGAP+1):
				video.set(1, x)
				frame = getOptFlowFrame(video)
				frame = cleanFrame(frame)
				frames.append(frame)

			frames = np.asarray(frames)

			if(not os.path.exists(label + "allvid-OPT")):
				os.makedirs(label + "allvid-OPT")
			np.save(label + "allvid-OPT/" + vid_name + str(framenum) + ".npy", frames)

	else:
		print("Already finished this sample.")
	# while(1):
	# 	ret, frame2 = cap.read()
	# 	if(not ret):
	# 	    break
	# 	next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

	# 	flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	# 	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	# 	hsv[...,0] = ang*180/np.pi/2


	# 	mag[mag > 0.005] = 0

	# 	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	# 	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	# 	video.write(rgb)


	# video.release()

	# cap.release()

	# cv2.destroyWindows()

	
	# return cv2.VideoCapture("temp_opt_flow.avi")




# def getOptFlowVid(vid_name):
# 	cap = cv2.VideoCapture("../vidsp2/" + vid_name)

# 	ret, frame1 = cap.read()
# 	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# 	hsv = np.zeros_like(frame1)
# 	hsv[...,1] = 255
# 	video = cv2.VideoWriter('temp_opt_flow.avi', cv2.VideoWriter_fourcc(*'XVID'),30,(1024,512))



# 	while(1):
# 		ret, frame2 = cap.read()
# 		if(not ret):
# 		    break
# 		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

# 		flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# 		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
# 		hsv[...,0] = ang*180/np.pi/2


# 		mag[mag > 0.005] = 0

# 		hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
# 		rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

# 		video.write(rgb)


# 	video.release()

# 	cap.release()

# 	# cv2.destroyWindows()

	
# 	return cv2.VideoCapture("temp_opt_flow.avi")



'''
Input: video from opencv in numpy array, 
vid_name as string for name of video to be saved as
label as string for class, 
framenum as frame where action occurs from csv

Process: Saves frame of video and context frames around it into pickled file
'''
def saveDataPoint(video, vid_name, label, framenum):

	if(not os.path.exists(label + "allvid-normal/" + vid_name + str(framenum) + ".npy")):

		frames = []
		
		length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

		if(framenum-FRAMECTX*(FRAMEGAP+1) < 0 or 
			framenum+FRAMECTX*(FRAMEGAP+1) > length):
			print("Out of range, skipping sample. \n")
		else:
			for x in range(framenum - FRAMECTX*(FRAMEGAP+1),
				framenum + FRAMECTX*(FRAMEGAP+1) + 1, FRAMEGAP+1):
				video.set(1, x)
				ret, frame = video.read()
				frame = cleanFrame(frame)
				frames.append(frame)

			frames = np.asarray(frames)

			if(not os.path.exists(label + "allvid-normal")):
				os.makedirs(label + "allvid-normal")
			np.save(label + "allvid-normal/" + vid_name + str(framenum) + ".npy", frames)

	else:
		print("Already finished this sample.")

'''
Input: frame as numpy array from opencv

Return: cleaned frame as numpy array
'''
def cleanFrame(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(frame, (100,100))
	return frame

'''
Input: csv file name path

Process: passes each row to readCSVRow()
'''
def readCSV(csv_filename):

	with open(csv_filename, newline='', encoding='utf-7') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')

		for row in reader:
			readCSVRow(row)


readCSV("gap_crossing_bhaipv_annotate.csv")



