import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load YOLOv8 model (in ONNX format)
model = YOLO("best.onnx")  # Replace with the correct model path

# Streamlit app title
st.title("SLD Device Detection Demo")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Open the image with PIL and display it
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to a format YOLO can work with (OpenCV)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference on the image
    results = model(img_bgr)  # Run inference directly on the image

    # Access the first result
    result = results[0]

    im = result.plot()

    # Convert the image back to RGB (for display with PIL or Streamlit)
    #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(im)

    # Display the image with detections
    st.image(output_image, caption="Detected Image", use_column_width=True)
