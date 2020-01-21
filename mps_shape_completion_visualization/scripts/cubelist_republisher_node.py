import rospy
from mps_shape_completion_visualization import cubelist_publisher



if __name__ == "__main__":
    rospy.init_node()
    cubelist_publisher(None, None)
