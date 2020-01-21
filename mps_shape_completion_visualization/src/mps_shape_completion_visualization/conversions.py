import rospy
from mps_shape_completion_msgs.msg import OccupancyStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


def binvox_to_occupancyStamped(bv):
    """
    INPUT: bv: DATATYPE: color
    """



def occupancyStamped_to_cubelist(occupancy_msg, color):
    """
    Takes an OccumancyStamped message with optional color and return a ros Marker as a cubelist
    INPUT: occupancy_msg: DATATYPE: OccupancyStamped
    INPUT: color: DATATYPE: std_msgs.msg.ColorRGBA
    OUTPUT: visualization_msgs.msg.Marker
    """

    if color is None:
        color = ColorRGBA()
        color.a = 1.0
        color.r = 0.5
        color.g = 0.5
        color.b = 0.5

    
    m = Marker()
    m.header.frame_id = "/base_frame"
    m.header.stamp = rospy.get_rostime()

    m.type = m.CUBE_LIST
    m.action = m.ADD
    m.pose.orientation.w = 1

    m.scale.x = 0.1
    m.scale.y = 0.1
    m.scale.z = 0.1

    m.color = color

    for i in range(10):
        p = Point()
        p.x = float(i)/10
        p.y = 0
        p.z = 0
        m.points.append(p)
    return m
