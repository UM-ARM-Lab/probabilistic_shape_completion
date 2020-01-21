#!/usr/bin/env python

import rospy
import numpy as np
from mps_shape_completion_visualization.conversions import occupancyStamped_to_cubelist
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

if __name__ == "__main__":
    rospy.init_node("shape_completion_cubelist_republisher")
    pub = rospy.Publisher("shape_completion_marker", Marker, queue_size=100)

    color = ColorRGBA()
    color.a = 1.0

    while not rospy.is_shutdown():
        color.r = np.random.random()
        color.g = np.random.random()
        color.b = np.random.random()
        
        pub.publish(occupancyStamped_to_cubelist(None, color))
        # cubelist_publisher(pub, None)
        rospy.sleep(0.5)
    # print("hi")
