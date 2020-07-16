from __future__ import print_function
import rospy
from rviz_text_selection_panel_msgs.msg import TextSelectionOptions
from std_msgs.msg import String
from shape_completion_training.utils import data_tools


def send_display_names_from_metadata(metadata, callback):
    names_list = [data_tools.get_unique_name(elem) for _, elem in metadata.enumerate()]
    return send_display_names(names_list, lambda ind, name: callback(metadata, ind, name))


def send_display_names(names_list, callback):
    """
    Sends a list of names to interact with rviz_text_selection_panel
    @param names_list: list of names
    @param callback: python callback function of form `def fn(ind, name)`
    @return:
    """
    options_pub = rospy.Publisher('shapenet_options', TextSelectionOptions, queue_size=1)
    selection_map = {}
    tso = TextSelectionOptions()
    for i, name in enumerate(names_list):
        selection_map[name] = i
        tso.options.append(name)
    i = 1
    while options_pub.get_num_connections() == 0:
        i += 1
        if i % 10 == 0:
            rospy.loginfo("Waiting for options publisher to connect to topic: {}".format('shapenet_options'))
        rospy.sleep(0.1)
    options_pub.publish(tso)

    def callback_fn(str_msg):
        return callback(selection_map[str_msg.data], str_msg)

    sub = rospy.Subscriber('/shapenet_selection', String, callback_fn)
    rospy.sleep(0.1)

    return sub
