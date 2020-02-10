import ast

import numpy as np

with open("analysis-dec-16-ss90-96-98.txt", "r") as f:

	frame_start_attempts = []
	frame_attempts_lengths = []
	count = 0

	num_attempts = 0
	for line in f.readlines():
		# print(line)
		items = line.split(" ")
		name = items[0]
		
		
		if("SS00090" in name):
			count += 1
			data = line[line.index("[")-1:line.index("]")+1].strip()
			print(data)
			if(len(data) > 2):
				for x in ast.literal_eval(data):
					frame_start_attempts.append(x[0])
					frame_attempts_lengths.append(x[1])
					num_attempts += 1
			else:
				num_attempts += 1

	print(frame_start_attempts)
	print(frame_attempts_lengths)

	print(np.mean(frame_start_attempts))
	print(np.std(frame_start_attempts))

	print(np.mean(frame_attempts_lengths))
	print(np.std(frame_attempts_lengths))

	print("avg attempts per frame: " + str(num_attempts/count))

	print("count: " + str(count))