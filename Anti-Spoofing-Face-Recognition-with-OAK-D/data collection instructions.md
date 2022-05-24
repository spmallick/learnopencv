# Instructions for data collection for spoof classification mod

## Install requirements

`pip install -r requirements.txt`

## Run "collect_data.py" script

`python collect_data.py`

## Controls

1. Press `f` key to save all the detected faces as "Real"
2. Press `s` key to save all the detected faces as "Spoofed"
3. Press `q` key to quit

## Catergories

1. Real Face: A real face is as the name suggests a sample of an actual human face.
2. Spoofed Face: A spoofed face can be a printed or digital image of a face.

## Things to keep in mind

1. Collect as many samples as you can and try to get the same number of "real" and "spoofed" samples.
2. Be careful not to be too close to the camera (the depth map might be invalid in that case). Keep checking that the disparity map output and make sure valid for the bounding box region.
3. Get variations in the captured frame by tilting your head and capturing from different angles.
4. You can also try different accesories like glasses, mask etc.
5. When saving samples of the spoofed face, tilt the digital image or bend the printed image to get variations.
