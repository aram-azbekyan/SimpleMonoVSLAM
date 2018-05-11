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
cap = cv2.VideoCapture(0)

try:
	while(True):
		# capture frame-by-frame
		ret, frame = cap.read()

		# undistort current frame
		undist_im = undistortFrame(frame, model, to_gray=True)

		if tr.nViews == 0:
			# if there is no views, add first frame to a list
			P = np.array([[1,0,0,0],
						  [0,1,0,0],
						  [0,0,1,0]]) # Identity matrix
			P = np.dot(model.K, P)
			tr.addView(undist_im, P)
		elif tr.nViews == 1:
			# if there is only one view, just track the pose
			kp_new, kp_old, matches = tr.matchImgs(undist_im, tr.frames[tr.nViews-1], show_matches=False)
			R, t = tr.calculatePose(kp_new, kp_old, model)
			tr.cur_R = R
			tr.cur_t = t
			P = np.append(tr.cur_R, tr.cur_t, axis=1)
			P = np.dot(model.K, P)
			tr.addView(undist_im, P)
		else:
			# trk new pose
			kp_new, kp_old, matches = tr.matchImgs(undist_im, tr.frames[tr.nViews-1], show_matches=False)
			R, t = tr.calculatePose(kp_new, kp_old, model)
			tr.cur_t = tr.cur_t + np.dot(tr.cur_R, t)
			tr.cur_R = np.dot(R, tr.cur_R)
			P = np.append(tr.cur_R, tr.cur_t, axis=1)
			P = np.dot(model.K, P)
			tr.addView(undist_im, P)
			# triangulate features in 3D-space
			prevP = tr.P[tr.nViews-1]
			X = mp.triangulateCoords(P, prevP, kp_new, kp_old)
			# update point-cloud
			pc.updatePc(X)

except Exception(e):
	print(e)



# release the capture
cap.release()
cv2.destroyAllWindows()