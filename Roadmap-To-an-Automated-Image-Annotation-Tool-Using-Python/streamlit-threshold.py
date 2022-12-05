import cv2 
import numpy as np 
import streamlit as st 
from PIL import Image
from io import BytesIO
import base64
from ads import css_string


def display_colorsp(images):
	cols = st.sidebar.columns(2)
	cols[0].text('Original')
	cols[0].image(images['Original'], channels="BGR")
	cols[1].text('Gray')
	cols[1].image(images['Gray'])
	cols[0].text('Blue')
	cols[0].image(images['Blue'])
	cols[1].text('Green')
	cols[1].image(images['Green'])
	cols[0].text('Red')
	cols[0].image(images['Red'])
	cols[1].text('Hue')
	cols[1].image(images['Hue'])
	cols[0].text('Saturation')
	cols[0].image(images['Saturation'])
	cols[1].text('Lightness')
	cols[1].image(images['Lightness'])


# Create application title and file uploader widget.
st.title("Thresholding using OpenCV")

exp = st.expander('', expanded=True)
col = exp.columns(2)

st.sidebar.header('COLORSPACES')
st.sidebar.header("STEPS")
st.sidebar.text("1️⃣ Upload the file.")
st.sidebar.text("2️⃣ Select a contrasting colorspace.")
st.sidebar.text("3️⃣ Set threshold type and adjust the slider.")

img_file_buffer = col[0].file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

thresh_type = col[1].selectbox('Select Threshold Type', ('Binary', 'Binary Inv', 'Trunc'))
colorsp = col[1].selectbox('Select Colorspace', ('Gray', 'Blue', 'Green', 
								   'Red', 'Hue', 'Saturation', 'Lightness'))

if img_file_buffer is not None:

	raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
	# Loads image in a BGR channel order.
	image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

	# Or use PIL Image (which uses an RGB channel order)
	# image = np.array(Image.open(img_file_buffer))

	# Create placeholders to display input and output images.
	placeholders = st.columns(2)
	# Display Input image in the first placeholder.
	placeholders[0].image(image, channels='BGR')
	placeholders[0].text("Input Image")

	# Create a Slider and get the threshold from the slider.
	threshold = st.slider("Threshold", min_value=0, max_value=255, step=1, value=127)

	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_blue = cv2.split(image)[0]
	img_green = cv2.split(image)[1]
	img_red = cv2.split(image)[2]
	img_hue = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[0]
	img_sat = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]
	img_val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

	img = {'Original': image, 'Gray': img_gray, 'Blue': img_blue, 'Green': img_green, 
		   'Red': img_red, 'Hue': img_hue, 'Saturation': img_sat, 'Lightness': img_val
	}

	display_colorsp(img)



	type_thresh = { 'Binary': cv2.THRESH_BINARY,
					'Binary Inv': cv2.THRESH_BINARY_INV,
					'Trunc': cv2.THRESH_TRUNC
	}
	
	ret, thresh = cv2.threshold(img[colorsp], threshold, 255, type_thresh[thresh_type])

	placeholders[1].image(thresh)
	placeholders[1].text("thresholded Image")

st.sidebar.text(" ")
st.sidebar.text(" ")

st.sidebar.markdown(css_string, unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 