import os
import random
import shutil

real_face_dir = os.path.join("dataset", "real")
os.makedirs(real_face_dir, exist_ok=True)
spoofed_face_dir = os.path.join("dataset", "spoofed")
os.makedirs(spoofed_face_dir, exist_ok=True)

training_real_dir = os.path.join("training_dataset", "real")
os.makedirs(training_real_dir, exist_ok=True)
test_real_dir = os.path.join("test_dataset", "real")
os.makedirs(test_real_dir, exist_ok=True)

training_spoof_dir = os.path.join("training_dataset", "spoofed")
os.makedirs(training_spoof_dir, exist_ok=True)
test_spoof_dir = os.path.join("test_dataset", "spoofed")
os.makedirs(test_spoof_dir, exist_ok=True)

_, _, real_list = next(os.walk(real_face_dir))
_, _, spoof_list = next(os.walk(spoofed_face_dir))

random.shuffle(real_list)
real_split_idx = int(len(real_list)*.10)
real_test = real_list[:real_split_idx]
real_train = real_list[real_split_idx:]
for file in real_test:
    shutil.copy(os.path.join(real_face_dir, file), test_real_dir)
for file in real_train:
    shutil.copy(os.path.join(real_face_dir, file), training_real_dir)

random.shuffle(spoof_list)
spoof_split_idx = int(len(spoof_list)*.10)
spoof_test = spoof_list[:spoof_split_idx]
spoof_train = spoof_list[spoof_split_idx:]
for file in spoof_test:
    shutil.copy(os.path.join(spoofed_face_dir, file), test_spoof_dir)
for file in spoof_train:
    shutil.copy(os.path.join(spoofed_face_dir, file), training_spoof_dir)
