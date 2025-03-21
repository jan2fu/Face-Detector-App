import cv2
import os


def test_cascade_classifier_initialization():
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    assert not classifier.empty(), "Failed to load haarcascade XML file"

def test_image_read():
    img = cv2.imread('DR.jpg')
    assert img is not None, "Image file could not be loaded"

def test_image_grayscale_conversion():
    img = cv2.imread('DR.jpg')
    grayscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert grayscaled_image is not None, "Image could not be converted to grayscale"
