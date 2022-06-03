import cv2
import numpy as np
from skimage.filters import roberts

def ToRGB(img):
	return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def ToGRAY(img):
	imgRBG = ToRGB(img)
	return (cv2.cvtColor(imgRBG, cv2.COLOR_BGR2GRAY))
#_______________________________________________Algorithme de filtrages__________________________________________
def median(img):
	imgGRAY = ToGRAY(img)
	return (cv2.medianBlur(imgGRAY, 9))

def gaussien(img):
	imgGRAY = ToGRAY(img)
	return (cv2.GaussianBlur(imgGRAY,(3,3),0))

def moyenneur(img):
	imgGRAY = ToGRAY(img)
	kernel = np.ones((3,3),np.float32)/9
	return (cv2.filter2D(imgGRAY,-1,kernel))
#_______________________________________________Algorithme de Segmentation______________________________________ 
def canny(img):
	return (cv2.Canny(img,100,200))

def sobel(img):
	sobel = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=1)
	return (sobel)

def perwit(img):
	kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	img_prewittx = cv2.filter2D(img, -1, kernelx)
	img_prewitty = cv2.filter2D(img, -1, kernely)
	return (img_prewittx + img_prewitty)

def laplacien(img):
	return (cv2.Laplacian(img, cv2.CV_8U, -1))

def roberts(img):
	return (roberts(img))
	
def findContours(img):
	_, binary = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
	image = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
	return (image)


def kmeans(img):
	pixel_values = img.reshape((-1, 3))
	pixel_values = np.float32(pixel_values)
	cond = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
	k = 5
	_, labels, (centers) = cv2.kmeans(pixel_values, k, None, cond, 10, cv2.KMEANS_RANDOM_CENTERS)
	centers = np.uint8(centers)
	labels = labels.flatten() 
	segmented_image = centers[labels.flatten()]
	segmented_image = segmented_image.reshape(img.shape)
	return segmented_image



#_______________________________________________Algorithme de Morphologie Mathématique_________________________ 

def dilatation(img):
	kernel = np.ones((3,3), np.uint8)
	return (cv2.dilate(img ,kernel,iterations = 1))

def erosion(img):
	kernel = np.ones((3,3), np.uint8)
	return (cv2.erode(img, kernel, iterations = 1))

def ouverture(img):
	kernel = np.ones((3,3),np.uint8)
	return (cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel))
def cloture(img):
	kernel = np.ones((3,3),np.uint8)
	return (cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel))

def TopHat(img) :
	kernel = np.ones((5,5),np.uint8)
	return (cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel))

def gradient(img):
	kernel = np.ones((5,5),np.uint8)
	return (cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel))

def BlackHat(img):
	kernel = np.ones((5,5),np.uint8)
	return (cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel))