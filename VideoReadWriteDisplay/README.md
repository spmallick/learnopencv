# [Read, Write, and Display a Video using OpenCV](https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/)

[<img src="https://learnopencv.com/wp-content/uploads/2017/05/The-Horse-in-Motion-anim.gif" alt="The Horse in Motion frames illustrate how still images form a video.">](https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/)

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/video-read-write-opencv-2026.07.27/VideoReadWriteDisplay-2026.07.27.zip)

This refreshed project shows checked, headless-by-default video reading and transcoding in Python 3 and C++17. Both programs accept explicit paths, reject invalid inputs, verify that the output writer opened, preserve measured frame size and frame rate by default, and print a JSON summary.

## Read a video

```bash
python -m pip install -r requirements.txt
python videoRead.py --input chaplin.mp4 --max-frames 30
```

Add `--display` when a desktop is available. Press `q` or Escape to stop the window.

## Transcode a video

```bash
python videoWrite.py --input chaplin.mp4 --output output.avi \
  --codec MJPG --max-frames 30
```

`codec` must contain four characters. Codec availability depends on the backend in your OpenCV build, so the program checks `VideoWriter.isOpened()` before writing.

## Build and run C++17

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
./build/video_read --input=chaplin.mp4 --max-frames=30
./build/video_write --input=chaplin.mp4 --output=output.avi \
  --codec=MJPG --max-frames=30
ctest --test-dir build --output-on-failure
```

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

The regression suite reads an exact frame count, checks the bundled clip's `640x360` geometry, transcodes eight frames, reopens the result, and exercises invalid-codec, invalid-frame-rate, and missing-file errors. It passed with Python OpenCV 4.13 and exact OpenCV 5.0.0, plus exact native C++ OpenCV 4.13.0 and exact 5.0.0.

---

<p align="center">
  <a href="https://bigvision.ai/">
    <img src="https://bigvision.ai/logos/logo.png" alt="BigVision.AI" width="300">
  </a>
</p>

<h2 align="center">Build Production-Ready Computer Vision &amp; AI Solutions</h2>

<p align="center">
  LearnOpenCV is maintained by <a href="https://bigvision.ai/"><strong>BigVision.AI</strong></a>, a computer vision and AI consulting company. We help organizations design, build, optimize, and deploy production-ready AI solutions. Our team has deep expertise in computer vision, deep learning, multimodal AI, and edge deployment, with experience solving complex technical challenges across industries.
</p>

<p align="center">
  Have a project in mind? Talk with our expert AI solution builders.
</p>

<p align="center">
  <a href="https://bigvision.ai/expert-ai-solution-builders?utm_source=locv-github">
    <img src="https://img.shields.io/badge/Get%20in%20Touch-087EA4?style=for-the-badge" alt="Get in Touch with BigVision.AI">
  </a>
</p>
