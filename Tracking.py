import cv2
import numpy as np

# initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# initiazize Brute-force matcher
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

class Tracking:
	"""
	Tracking class carries out all processes and data
	that belong to tracking step in vslam
	"""
	def __init__(self):
		self.nViews = 0
		self.P = {}
		self.frames = {}
		self.cur_t = None
		self.cur_R = None

	def addView(self, im, newP):
		""" Add new view to a list. """
		self.P[self.nViews] = newP
		self.frames[self.nViews] = im
		self.nViews += 1

	def matchImgs(self, newFrame, prevFrame, show_matches=False):
		""" Match two frames and return matched keypoints. """

		# first, take off keypoints
		kp1, des1 = orb.detectAndCompute(newFrame, None)
		kp2, des2 = orb.detectAndCompute(prevFrame, None)

		# then, get matches
		matches = matcher.match(des1, des2)
		# take only reliable matches
		matches = [m for m in matches if m.distance < 60]
		# sort list by preciseness
		matches = sorted(matches, key=lambda x:x.distance)

		# draw matches
		if show_matches:
			im = cv2.drawMatches(newFrame, kp1, prevFrame, kp2, matches, None, flags=2)
			cv2.imshow('Matches', im)
			cv2.waitKey(1)

		kp1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
		kp2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

		return kp1, kp2, matches

	def calculatePose(self, kp_new, kp_old, camera):

		# find essental matrix
		E, mask = cv2.findEssentialMat(kp_new, kp_old, focal=camera.fx, pp=camera.pp, method=cv2.RANSAC, prob=0.9999, threshold=1.0)

		# recover pose
		points, R, t, mask = cv2.recoverPose(E, kp_new, kp_old, focal=camera.fx, pp=camera.pp)

		return R, t