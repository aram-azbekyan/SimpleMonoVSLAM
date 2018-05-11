import cv2
import numpy as np

class Mapping:
	""" 
	Mapping class performs operations to evaluate
	and store 3D features
	"""
	def __init__(self):
		self.X = []
		
	def triangulateCoords(self, P_new, P_old, kp_new, kp_old):
		"""
		take projection matricies and key-points
		and calculate 3D-coordinates of them
		"""
		self.X = cv2.triangulatePoints(P_new, P_old, kp_new, kp_old)
		return self.X