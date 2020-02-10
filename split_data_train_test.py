#Run only once to sort out data
import os, random
from shutil import copy

attempt_path = "data-normal-v1\\attempt-total"
normal_path = "data-normal-v1\\nothing-total"

dst = "data-normal-v1"

attempt_dir = os.listdir(attempt_path)
normal_dir = os.listdir(normal_path)


attempt_train = random.sample(attempt_dir, len(attempt_dir))
normal_train = random.sample(normal_dir, len(normal_dir))

attempt_test = random.sample(attempt_dir, int(len(attempt_dir)/5))
normal_test = random.sample(normal_dir, int(len(normal_dir)/5))

attempt_train = list(set(attempt_train).difference(set(attempt_test)))
normal_train = list(set(normal_train).difference(set(normal_test)))

for x in attempt_train:
    copy(attempt_path + '\\' + x, dst + '\\train\\attempt\\' + x)

for x in attempt_test:
    copy(attempt_path + '\\' + x, dst + '\\test\\attempt\\' + x)

for x in normal_train:
    copy(normal_path + '\\' + x, dst + '\\train\\nothing\\' + x)

for x in normal_test:
    copy(normal_path + '\\' + x, dst + '\\test\\nothing\\' + x)
