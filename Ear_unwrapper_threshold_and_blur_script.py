#Script to process unwrapped images with pixel probabilities

import cv2
import numpy as np
import sys

if len(sys.argv) < 1:
    print("No input file specified")
    exit()

#use the command line argument as the image name
input_img = sys.argv[1]

#read in the image
img = cv2.imread(input_img)

# Convert BGR to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Initialize our arrays for thresholding
lower_blue = np.array([80,180,120])
upper_blue = np.array([120,255,255])
height, width, _ = hsv_img.shape

#Crop the right side pixels of empty space
cropped_img = hsv_img[0:height, 0:(width-50)]

#Perform a median Blur
blurred_img = cv2.medianBlur(cropped_img, 11)

# Threshold the HSV image for only blue color, and create a binary (black and white) mask 
mask = cv2.inRange(blurred_img, lower_blue, upper_blue)

#Output the binary mask
output_name = "medianblur-" + input_img
cv2.imwrite(output_name, mask)