import cv2
import imutils
import numpy as np
import pytesseract  
from PIL import Image, ImageFilter

def performOCR(image):
    resized_image = cv2.resize(image,(600, 400))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(grayscale_image, 13, 17, 17)

    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection 30,200
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)

    # arrange contours descending based on contour area and get largest 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  

    image1 = resized_image.copy()

    # drawing the identified contours on our image
    cv2.drawContours(image1,contours,-1,(100,255,0),3)

    plate_location = None
    for contour in contours:
        epsilon =  0.018*cv2.arcLength(contour, True)  # Ratio of contour Perimeter 
        approx = cv2.approxPolyDP(contour, epsilon, True)  # approximate contour shape 
    
        if len(approx) == 4:
            plate_location = approx
            break

    # drawing the identified contours on our image
    cv2.drawContours(resized_image,[plate_location],-1,(100,255,0),3)
    
    mask = np.zeros(grayscale_image.shape,np.uint8)
    new_image = cv2.drawContours(mask,[plate_location],0,255,-1,)
    new_image = cv2.bitwise_and(resized_image,resized_image,mask=mask)

    (x,y) = np.where(mask==255)
    (x1,y1) = (np.min(x), np.min(y))
    (x2,y2) = (np.max(x), np.max(y))
    cropped_image = grayscale_image[x1:x2+1, y1:y2+1]

    # apply thresholding to improve image quality
    _, thresh = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # recognize text from the image
    text = pytesseract.image_to_string(cropped_image, lang='eng', config='--psm 7')

    return text