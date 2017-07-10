import numpy as np
import cv2

# 3D color histogram
# 4 bins for Hue channel, 6 bins for saturation channel, 3 bins for value channel
# => feature vector of dimensiont 4 x 6 x 3 = 72

# use regions-based histograms (divide image into 5 different regions)
# rather than global-historgrams allow to simulate locality in color distribution


class HSVDescriptor:
	def __init__(self, bins):
		# store the number of bins for the histogram
		self.bins = bins

	def histogram(self, image, mask = None):
		# extract 3D color histogram from the masked region of the image
		# using supplied number of bins per channel
		# then normalize the histogram		
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,\
							[0, 180, 0, 256, 0, 256])
		hist = cv2.normalize(hist).flatten()

		# return the histogram
		return hist

	def describe(self, image):
		# convert the iamge to the HSV color space and initialize
		# the features used to quantify the image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		# grab the dimensions and compute the center of the image
		h, w = image.shape[:2]
		cX, cY = int(w * 0.5), int(h * 0.5)

		# divide the image into 4 segments
		# top-left, top-right, bottom-right, bottom-left)
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

		# construct an elliptical mask representing center of the image
		axesX, axesY = int(w * 0.75) / 2, int(h * 0.75) / 2
		ellipMask = np.zeros(image.shape[:2], dtype = 'uint8')
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		# loop over the segments
		for startX, endX, startY, endY in segments:
			# construct a mask for each corner of the image, 
			# subtracting the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = 'uint8')
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)

			# extract a color histogram from the image, then update feature vector
			hist = self.histogram(image, cornerMask)
			features.extend(hist)

		# extract color histogram from the elliptical region and update feature vector
		hist = self.histogram(image, ellipMask)
		features.extend(hist)

		# return feature vector
		return np.array(features)
