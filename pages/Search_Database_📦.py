import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from Main_Operations import performOCR
import time
import re

st.sidebar.markdown('Search Car License Plate 🔍')

uploaded_image = st.file_uploader("# Upload the vehicle's image", type=["jpg", "jpeg", "png"])

license = None

# Load the dataset
df = pd.read_csv("License_Plate_Database.csv")

# Function to check if the input string matches a license number
def check_license(license):
    matching_row = df[df['License Number'] == license]
    if not matching_row.empty:
        car_name = matching_row.iloc[0]['Car Name']
        st.success(f"\n## Match found: \n#### Car Name - {car_name}\n#### License Number - {license}")

    else:
        st.error("\n### No match found for the given license number.")
        


if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, use_column_width=True)

    if st.button("Search 🔍"):
            # Convert the uploaded image to a format usable by OpenCV
            img_array = np.array(image)
            opencv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Perform image preprocessing
            license = performOCR(opencv_image)
            license = re.sub(r'[^a-zA-Z0-9]', '', license)

            with st.status("#### Searching for the license number...", expanded=True):
                st.write("##### Applying filters...")
                time.sleep(1)
                st.write("##### Performing OCR")
                time.sleep(1)

            check_license(license)

