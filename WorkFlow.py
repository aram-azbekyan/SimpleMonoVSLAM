import cv2
import numpy as np
import signal
import sys

import rospy

from Camera import *
from Tracking import *
from Mapping import *
from PointCloud import *

def signal_handler(signal, frame):
	print('Ctrl-C pressed')
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
print('Ctrl-C to close program')

# initializations
model = Camera()  # camera model
tr = Tracking()     # tracking module
mp = Mapping()    # mapping module
pc = PointCloud() # point cloud
rospy.init_node('cloud_stream', anonymous=True)

# video capture object
cap = cv2.VideoCapture(1)

# nframes = 0

while(True):
	# capture frame-by-frame
	ret, frame = cap.read()

	# nframes += 1
	# if nframes % 10 != 0:
	# 	continue

	undist_im = undistortFrame(frame, model, to_gray=True)

	if tr.nViews == 0:
		# first view incoming
		Rt = np.array([[1,0,0,0],
					  [0,1,0,0],
					  [0,0,1,0]]) # Identity matrix
		P = np.dot(model.K, Rt)
		tr.addView(undist_im, P)
	elif tr.nViews == 1:
		kp_new, kp_old, matches = tr.matchImgs(undist_im, tr.frames[tr.nViews-1], show_matches=False)
		R, t = tr.calculatePose(kp_new, kp_old, model)
		tr.cur_t = 0.01 * np.dot(-R.T,t)
		tr.cur_R = R.T
		Rt = np.append(tr.cur_R, tr.cur_t, axis=1)
		P = np.dot(model.K,Rt)
		tr.addView(undist_im, P)
	else:
		# track new pose
		kp_new, kp_old, matches = tr.matchImgs(undist_im, tr.frames[tr.nViews-1], show_matches=False)
		R, t = tr.calculatePose(kp_new, kp_old, model)
		tr.cur_t = tr.cur_t + 0.01 * np.dot(-R.T,t)
		tr.cur_R = np.dot(R.T, tr.cur_R)
		Rt = np.append(tr.cur_R, tr.cur_t, axis=1)
		P = np.dot(model.K,Rt)
		tr.addView(undist_im, Rt)
		rvec = cv2.Rodrigues(Rt[:,:-1])
		tvec = Rt[:,-1:]
		tr.handle_camera_pose(rvec,tvec)
		# # triangulate features in 3D-space
		# X = mp.triangulateCoords(Rt, lastRt, model, kp_new, kp_old)
		# # update point-cloud
		# pc.updatePc(X)



# release the capture
cap.release()
cv2.destroyAllWindows()