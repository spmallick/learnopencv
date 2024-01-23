import os
import shutil

prepend = 'PATH_TO_SCENEFLOW/SceneFlow/FlyingThings3D/frame_finalpass'
unused_dir = os.path.join(prepend, 'UNUSED')
if not os.path.isdir(unused_dir):
    os.mkdir(unused_dir)
if not os.path.isdir(os.path.join(unused_dir, 'TRAIN')):
    os.mkdir(os.path.join(unused_dir, 'TRAIN'))
if not os.path.isdir(os.path.join(unused_dir, 'TEST')):
    os.mkdir(os.path.join(unused_dir, 'TEST'))
if not os.path.isdir(os.path.join(unused_dir, 'TRAIN/A')):
    os.mkdir(os.path.join(unused_dir, 'TRAIN/A'))
if not os.path.isdir(os.path.join(unused_dir, 'TRAIN/B')):
    os.mkdir(os.path.join(unused_dir, 'TRAIN/B'))
if not os.path.isdir(os.path.join(unused_dir, 'TEST/A')):
    os.mkdir(os.path.join(unused_dir, 'TEST/A'))
if not os.path.isdir(os.path.join(unused_dir, 'TEST/B')):
    os.mkdir(os.path.join(unused_dir, 'TEST/B'))

# download unused files at link: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlow/assets/all_unused_files.txt
f = open('PATH_TO_SCENEFLOW/all_unused_files.txt', 'r')
for img in f:
    img = img.strip().rstrip()
    img_fname = os.path.join(prepend, img)

    if os.path.isfile(img_fname):
        print(img_fname)
        target = os.path.join(unused_dir, img)
        print(target)
        # adsf
        if not os.path.isdir(target[:-14]):
            os.mkdir(target[:-14])
        if not os.path.isdir(target[:-9]):
            os.mkdir(target[:-9])
        shutil.move(img_fname, target)
