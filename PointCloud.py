import numpy as np
import rospy
from roslib import message
import std_msgs.msg
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2

class PointCloud:
	"""
	PointCloud class generates and sends
	point cloud data to RViz.
	"""
	def __init__(self):
		self.pc_pub = rospy.Publisher('cloud_stream',
					  PointCloud2, queue_size=10)

	def updatePc(self, pts3d):
		""" Process 3d points obtained from triangulation step
		and send them in RViz. """
		
		# formulate message header
		self.h = std_msgs.msg.Header()
		self.h.stamp = rospy.Time.now()
		self.h.frame_id = 'map'

		# eliminate 4th (W) coordinate and compile a message
		cloud = pcl2.create_cloud_xyz32(self.h, pts3d)
		self.pc_pub.publish(cloud)