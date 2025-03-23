import cv2
import os
from random import randrange

# Load some pretrained data on face frontals from opencv (haarcascade algorithm)
# The classifier is an algorithm that detects objects based on features
# It is trained on a lot of images of faces and non-faces
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# Call the imread function to read the image
# img = cv2.imread('crowd1.jpg')

# Call the video capture function to capture video from the webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()
    
    # Convert the image to grayscale
    # cvtColor function converts the image to a different color space
    # The first parameter is the image to convert and the second parameter is the color space to convert to grayscale
    grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # The first parameter is the image to detect objects in and the second parameter is the scale factor
    # detectMultiScale function detects objects in the image
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)

    # Draw rectangles around the faces
    # rectangle function draws a rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+h, y+h), (randrange(256)), 2)

# print(face_coordinates)

    # Call imshow function to display the image
    # cv2.imshow('Face Detector App', img) 
    # WaitKey function pauses the program and waits for a key press to close the window
    # adding the waitKey function with a parameter of 1 will display the image for 1 millisecond
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

    # Release the VideoCapture object
    webcam.release()




print ("Code Completed")