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
ds_cam = DatasetCamera()
tr = Tracking()     # tracking module
mp = Mapping()    # mapping module
pc = PointCloud() # point cloud
rospy.init_node('cloud_stream', anonymous=True)

# video capture object
# cap = cv2.VideoCapture(0)

for im_id in xrange(108):

	im = cv2.imread('./dataset/image_00/data/'+str(im_id).zfill(10)+'.png', 0)

	if im_id == 0:
		tr.kp_old = detector.detect(im)
		tr.kp_old = np.array([ x.pt for x in tr.kp_old], dtype=np.float32)
		tr.addView(im)
	elif im_id == 1:
		tr.featureTracking(im)
		tr.cur_R, tr.cur_t = tr.calculatePose(ds_cam)
		tr.cur_t = tr.cur_t * 0.2
		tr.kp_old = tr.kp_new
		tr.lastRt = np.append(tr.cur_R, tr.cur_t, axis=1)
		tr.addView(im)
	else:
		tr.featureTracking(im)
		R, t = tr.calculatePose(ds_cam)
		tr.cur_t = tr.cur_t + 0.2 * tr.cur_R.dot(t)
		tr.cur_R = R.dot(tr.cur_R)
		rvec = cv2.Rodrigues(tr.cur_R)
		tr.handle_camera_pose(tr.cur_t, rvec)
		Rt = np.append(tr.cur_R, tr.cur_t, axis=1)
		X = mp.triangulateCoords(Rt, tr.lastRt, ds_cam, tr.kp_new.T, tr.kp_old.T)
		pc.updatePc(X)
		tr.kp_old = detector.detect(im)
		tr.kp_old = np.array([ x.pt for x in tr.kp_old], dtype=np.float32)
		tr.addView(im)



# release the capture
cap.release()
cv2.destroyAllWindows()