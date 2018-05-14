import cv2
import numpy as np

class Mapping:
	""" 
	Mapping class performs operations to evaluate
	and store 3D features
	"""
	def __init__(self):
		self.X = []
		
	def triangulateCoords(self, P_new, P_old, cam, kp_new, kp_old):
		"""
		take projection matricies and key-points
		and calculate 3D-coordinates of them
		"""

		# undistort key-points
		# normalizedNew = cv2.undistortPoints(kp_new, cam.K, cam.distortion)
		# normalizedOld = cv2.undistortPoints(kp_old, cam.K, cam.distortion)

		# triangulate 3D-coordinates of matches points
		X_homog = cv2.triangulatePoints(P_new, P_old, kp_new, kp_old)

		# make transition from homogeneous coordinates into Euclidean
		newX = cv2.convertPointsFromHomogeneous(X_homog.T)
		newX = [ x[0].tolist() for x in newX ]
		# self.X.extend(newX)

		return newX