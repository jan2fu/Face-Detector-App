import cv2
import os

# Load some pretrained data on face frontals from opencv (haarcascade algorithm)
# The classifier is an algorithm that detects objects based on features
# It is trained on a lot of images of faces and non-faces
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# Call the imread function to read the image
img = cv2.imread('DR.jpg')

# Convert the image to grayscale
# cvtColor function converts the image to a different color space
# The first parameter is the image to convert and the second parameter is the color space to convert to grayscale
grayscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Call imshow function to display the image
cv2.imshow('Face Detector App', grayscaled_image)
# WaitKey function pauses the program and waits for a key press to close the window
cv2.waitKey()




print ("Code Completed")