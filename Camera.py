import cv2
import numpy as np

class Camera:
	""" Camera object. """
	def __init__(self):
		self.fx = 601.605896
		self.fy = 598.708191
		self.cx = 320.0
		self.cy = 240.0
		self.pp = (self.cx, self.cy)
		self.K = np.array([[self.fx,   0,         self.cx],
						   [0,         self.fy,   self.cy],
						   [0,         0,         1]])
		self.distortion = np.array([[0.128536, -0.356942, -0.003803, 0.000213, 0.394380]])
		self.roi = (2, 2, 636, 476)

class DatasetCamera:
	def __init__(self):
		self.fx = 718.8560
		self.fy = 718.8560
		self.cx = 607.1928
		self.cy = 185.2157
		self.pp = (self.cx, self.cy)
		self.K = np.array([[self.fx,   0,         self.cx],
						   [0,         self.fy,   self.cy],
						   [0,         0,         1]])
		self.distortion = [0.0, 0.0, 0.0, 0.0, 0.0]

def undistortFrame(im, cam, to_gray=False):
	""" Take new frame and undistort it. """
	undist_im = cv2.undistort(im, cameraMatrix=cam.K, distCoeffs=cam.distortion)
	x,y,w,h = cam.roi
	cropped_im = undist_im[y:y+h, x:x+w]
	if to_gray:
		cropped_im = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2GRAY)
	return cropped_im