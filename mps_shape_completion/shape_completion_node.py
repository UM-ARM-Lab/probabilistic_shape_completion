#!/usr/bin/env python

from shape_complete import ShapeCompleter
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from mps_shape_completion_msgs.srv import CompleteShape, CompleteShapeRequest, CompleteShapeResponse
from rospy.numpy_msg import numpy_msg

from shape_utils import vox_to_msg, msg_to_vox
import rospy
import rospkg

from datetime import datetime


def service_callback(req, args):
    sc = args

    arr = msg_to_vox(req.observation)

    occ = arr >= 0.5
    non = arr < 0.5

    time_stamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    sc.save_input(occ, non, '.', time_stamp)
    out = sc.complete(occ=occ, non=non, verbose=False, save=True, id=time_stamp, out_path='.')

    resp = CompleteShapeResponse()
    resp.hypothesis = vox_to_msg(out)
    return resp


def callback(msg, args):
    """
    Callback for handling CompleteShape requests.
    msg: the input voxel grid
    args: (ShapeCompleter, PredictedOccupancyPublisher)
    """
    sc = args[0]
    pub = args[1]

    arr = msg_to_vox(msg)

    occ = arr > 0
    non = arr < 0

    out = sc.complete(occ=occ, non=non, verbose=False)

    pub.publish(vox_to_msg(out))


def listener():
    rospy.init_node('shape_completer')

    model_path = rospkg.RosPack().get_path('mps_shape_completion') + "/train_mod/"
    sc = ShapeCompleter(model_path, verbose=True)

    server = rospy.Service('complete_shape', CompleteShape, lambda msg: service_callback(msg, sc))

    pub = rospy.Publisher('local_occupancy_predicted', numpy_msg(Float32MultiArray), queue_size=10)
    rospy.Subscriber("local_occupancy", numpy_msg(Float32MultiArray), callback, (sc, pub))
    rospy.loginfo("Shape completer ready")
    rospy.spin()


if __name__ == '__main__':
    listener()
