import cv2
import numpy as np 
import streamlit as st
from ads import css_string


def segment_color(img, lb, ub):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	color_mask = cv2.inRange(hsv_img, np.array(lb), np.array(ub))
	return color_mask


# Create application title and file uploader widget.
st.title("Color Segmentation using OpenCV")

buffer = st.file_uploader('Select Image', type=['jpg', 'jpeg', 'png'])

st.sidebar.header('STEPS:')
st.sidebar.text("1️⃣ Upload image")
st.sidebar.text("2️⃣ Select color")
st.sidebar.text("3️⃣ Adjust sliders")
st.sidebar.text(" ")

st.sidebar.markdown(css_string, unsafe_allow_html=True)

upper_bound = []
lower_bound = []

line_hsv = cv2.imread('hsv-line.png')

if buffer is not None:
	raw_bytes = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
	# Loads image in a BGR channel order.
	image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

	# Create placeholders to display input and output images.
	placeholders = st.columns(2)
	
	# Display Input image in the first placeholder.
	placeholders[0].image(image, channels='BGR')
	placeholders[0].text("Input Image")

	st.image(line_hsv, channels='BGR')

	hue = st.slider('Hue', 0, 255, [0, 40])
	sat = st.slider('Saturation', 0, 255, [70, 255])
	val = st.slider('Lightness', 0, 255, [70, 255])

	lower_bound = [hue[0], sat[0], val[0]]
	upper_bound = [hue[1], sat[1], val[1]]


	mask = segment_color(image, lower_bound, upper_bound)

	# Display mask.
	placeholders[1].image(mask)
	placeholders[1].text('Color Mask')

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 