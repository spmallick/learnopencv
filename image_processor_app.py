import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Image Processor", layout="wide")
st.title("OpenCV Image Processing Tool")
st.write("This tool demonstrates various image processing techniques using OpenCV and Python.")

# Sidebar for file upload and processing options
with st.sidebar:
    st.header("Image Processing Options")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Processing options
        processing_type = st.selectbox(
            "Select Processing Type",
            ["Original", "Grayscale", "Edge Detection", "Blur", "Histogram Equalization"]
        )

# Main content area
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_column_width=True)
    
    with col2:
        st.subheader(f"Processed Image - {processing_type}")
        
        if processing_type == "Original":
            st.image(uploaded_file, use_column_width=True)
        
        elif processing_type == "Grayscale":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            st.image(gray, use_column_width=True, clamp=True)
        
        elif processing_type == "Edge Detection":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            st.image(edges, use_column_width=True, clamp=True)
        
        elif processing_type == "Blur":
            blurred = cv2.GaussianBlur(img_array, (15, 15), 0)
            st.image(blurred, use_column_width=True)
        
        elif processing_type == "Histogram Equalization":
            if len(img_array.shape) == 3:
                # Convert to HSV and equalize V channel
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                v_eq = cv2.equalizeHist(v)
                hsv_eq = cv2.merge((h, s, v_eq))
                result = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)
            else:
                result = cv2.equalizeHist(img_array)
            st.image(result, use_column_width=True)
else:
    st.info("👈 Upload an image to get started")

st.markdown("---")
st.write("Built with Streamlit and OpenCV - Learn OpenCV Contribution")
