import rospy
from mps_shape_completion_msgs.msgs import OccupancyStamped.msg



def cubelist_publisher(publisher, occupancy_msg):
    print("Called cubelist publisher")
