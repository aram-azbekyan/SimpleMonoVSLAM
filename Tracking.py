import cv2
import numpy as np

import rospy
import tf
import tf2_ros
import geometry_msgs.msg

# initialize ORB detector
orb = cv2.ORB_create(nfeatures=5000)

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
		self.br = tf2_ros.TransformBroadcaster()

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

	def handle_camera_pose(self, rvec, tvec):
		t = geometry_msgs.msg.TransformStamped()
		t.header.stamp = rospy.Time.now()
		t.header.frame_id = "map"
		t.child_frame_id = "camera_pose"
		t.transform.translation.x = tvec[0]
		t.transform.translation.y = tvec[2]
		t.transform.translation.z = tvec[1]
		# q = tf.transformations.quaternion_from_euler(rvec[0][0], rvec[0][1], rvec[0][2])
		t.transform.rotation.x = 0
		t.transform.rotation.y = 0
		t.transform.rotation.z = 0
		t.transform.rotation.w = 1
		self.br.sendTransform(t)