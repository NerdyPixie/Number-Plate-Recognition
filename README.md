# Number Plate Recognition Application

This Streamlit project is designed to perform number plate recognition using OpenCV and Tesseract. The application takes an image as input, enhances it using OpenCV techniques, extracts the number plate from the image, and then reads the text from the number plate using Tesseract OCR.  It also provides functionality to search the car number in the current database.


## Images

![image](https://github.com/NerdyPixie/Number-Plate-Recognition/assets/66908638/53448475-c83d-4f59-b7a9-d64f47931a94)

![image](https://github.com/NerdyPixie/Number-Plate-Recognition/assets/66908638/fb1ab70c-b487-48ca-a1fc-7a1467dd0415)



## Installation

1. Clone this repository:

git clone <https://github.com/NerdyPixie/Number-Plate-Recognition>


2. Install the required dependencies:

pip install -r requirements.txt



## Usage

1. Run the Streamlit application:

streamlit run Number_Plate_Recognition.py


2. Upload an image containing a vehicle with a visible number plate.
   

3. The application will process the image, identify the number plate, and display the extracted text.




## Code Explanation

The main functionality of the application is contained within the `performOCR` function. Here's a brief explanation of the key steps:

1. Load the image and preprocess it:
   - Resize the image for better processing.
   - Convert the image to grayscale.
   - Apply bilateral filter to reduce noise.

2. Detect edges in the image using Canny edge detection.

3. Find contours in the image and identify the largest contour assumed to be the number plate.

4. Draw the identified contour on the image and extract the region of interest (number plate).

5. Apply thresholding to the number plate to improve text recognition.

6. Use Tesseract to extract text from the number plate region.

7. Provide functionality to search the extracted car number in the current database.

## Dependencies

- OpenCV
- imutils
- numpy
- pytesseract
- streamlit
- Pillow

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests to improve the project.
