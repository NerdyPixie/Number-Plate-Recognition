import streamlit as st
import cv2
import numpy as np
from PIL import Image
from Main_Operations import performOCR
import re

st. set_page_config(layout="wide")
st.markdown("# License Plate Recognition 🚗")
st.sidebar.markdown("# Main page 🎈")


def main(): 
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Perform OCR on the image"):
            # Convert the uploaded image to a format usable by OpenCV
            img_array = np.array(image)
            opencv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Perform image preprocessing
            output = performOCR(opencv_image)
            output = re.sub(r'[^a-zA-Z0-9]', '', output)
            st.write("# The number on the plate is : " , output)



if __name__ == "__main__":
    main()