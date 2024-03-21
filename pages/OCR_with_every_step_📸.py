import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFilter
import imutils
import pytesseract
import re

st.markdown("# OCR with every step ")
st.sidebar.markdown("# :camera_with_flash: OCR with every step ")

# Image preprocessing
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    new_image = Image.open(uploaded_image)
    st.image(new_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Resize Image"):
        new_image = np.array(new_image)
        resized_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        new_image = cv2.resize(new_image, (600, 400))
        st.image(new_image, caption="Resized Image", use_column_width=True)
    

    if st.button("Convert to GrayScale"):
        new_image = np.array(new_image)
        resized_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(resized_image, (600, 400))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        st.image(gray, caption="Gray Image", use_column_width=True)
   

    if st.button("Bilateral Filter"):
        new_image = np.array(new_image)
        resized_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(resized_image, (600, 400))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        bfilter = cv2.bilateralFilter(gray, 13, 15, 15)
        st.image(bfilter, caption="Bilateral Filtered Image", use_column_width=True)
       

    if st.button("Detect Edges"):
        new_image = np.array(new_image)
        resized_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(resized_image, (600, 400))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        bfilter = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(bfilter, 30, 200)
        st.image(edged, caption="Edge Detected Image", use_column_width=True)

    if st.button("Find Contours"):
        new_image = np.array(new_image)
        resized_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(resized_image, (600, 400))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        bfilter = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(bfilter, 30, 200)
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        image1 = resized_image.copy()
        cv2.drawContours(image1,contours,-1,(100,255,0),3)
        st.image(image1, caption="Contoured Image", use_column_width=True)

    if st.button("Find number plate Contour"):
        new_image = np.array(new_image)

        resized_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(resized_image, (600, 400))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        bfilter = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        plate_location = None
        for contour in contours:
            epsilon =  0.018*cv2.arcLength(contour, True)  # Ratio of contour Perimeter 
            approx = cv2.approxPolyDP(contour, epsilon, True)  # approximate contour shape 
  
            if len(approx) == 4:
                plate_location = approx
                break

        # drawing the identified contours on our image
        cv2.drawContours(resized_image,[plate_location],-1,(100,255,0),3)
        st.image(resized_image, caption="Number Plate Contour", use_column_width=True)


    if st.button("Apply mask"):
        new_image = np.array(new_image)

        resized_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(resized_image, (600, 400))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        bfilter = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        plate_location = None
        for contour in contours:
            epsilon =  0.018*cv2.arcLength(contour, True)  # Ratio of contour Perimeter 
            approx = cv2.approxPolyDP(contour, epsilon, True)  # approximate contour shape 
  
            if len(approx) == 4:
                plate_location = approx
                break
   
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[plate_location],0,255,-1,)
        new_image = cv2.bitwise_and(resized_image,resized_image,mask=mask)
        st.image(new_image, caption="Masked Image", use_column_width=True)

    if st.button("Crop the number plate and apply thresholding"):
        new_image = np.array(new_image)

        resized_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(resized_image, (600, 400))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        bfilter = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        plate_location = None
        for contour in contours:
            epsilon =  0.018*cv2.arcLength(contour, True)  # Ratio of contour Perimeter 
            approx = cv2.approxPolyDP(contour, epsilon, True)  # approximate contour shape 
  
            if len(approx) == 4:
                plate_location = approx
                break
   
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[plate_location],0,255,-1,)
        new_image = cv2.bitwise_and(resized_image,resized_image,mask=mask)

        (x,y) = np.where(mask==255)
        (x1,y1) = (np.min(x), np.min(y))
        (x2,y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]

        # apply thresholding to improve image quality
        _, thresh = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        st.image(thresh, caption="Cropped Number plate", use_column_width=True)
        
    if st.button("Recognize text from the image"):
        new_image = np.array(new_image)

        resized_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(resized_image, (600, 400))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        bfilter = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        plate_location = None
        for contour in contours:
            epsilon =  0.018*cv2.arcLength(contour, True)  # Ratio of contour Perimeter 
            approx = cv2.approxPolyDP(contour, epsilon, True)  # approximate contour shape 
  
            if len(approx) == 4:
                plate_location = approx
                break
   
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[plate_location],0,255,-1,)
        new_image = cv2.bitwise_and(resized_image, resized_image,mask=mask)

        (x,y) = np.where(mask==255)
        (x1,y1) = (np.min(x), np.min(y))
        (x2,y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]

        # apply thresholding to improve image quality
        _, thresh = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Perform OCR on the corrected image
        text = pytesseract.image_to_string(cropped_image, lang='eng', config='--psm 7')
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        st.write('### The license plate number is : ', text)
