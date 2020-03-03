#!/usr/bin/env python
from __future__ import print_function


import rospy
import tf2_ros
import tf_conversions
import geometry_msgs.msg
from mps_shape_completion_msgs.msg import OccupancyStamped
from mps_shape_completion_visualization import conversions


import numpy as np

import sys
import os

sc_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(sc_path)
from shape_completion_training.model.network import AutoEncoderWrapper
from shape_completion_training.model import data_tools
from shape_completion_training.model import obj_tools
from shape_completion_training import binvox_rw

from bsaund_shape_completion import sampling_tools

from rviz_text_selection_panel_msgs.msg import TextSelectionOptions
from std_msgs.msg import String

import threading

import IPython


SAMPLING = True



# DIM = 64

gt_pub = None
known_occ_pub = None
known_free_pub = None
completion_pub = None
completion_free_pub = None
mismatch_pub = None

options_pub = None
selected_sub = None

model = None

selection_map = {}
stop_current_sampler = None
sampling_thread = None




def to_msg(voxel_grid):
    
    return conversions.vox_to_occupancy_stamped(voxel_grid,
                                                dim = voxel_grid.shape[1],
                                                scale=0.01,
                                                frame_id = "object")

def publish_elem(elem):
    gt_pub.publish(to_msg(elem["gt_occ"].numpy()))
    known_occ_pub.publish(to_msg(elem["known_occ"].numpy()))
    known_free_pub.publish(to_msg(elem['known_free'].numpy()))
    sys.stdout.write('\033[2K\033[1G')
    print("Category: {}, id: {}, aug: {}".format(elem['shape_category'].numpy(),
                                                 elem['id'].numpy(),
                                                 elem['augmentation'].numpy()), end="")
    sys.stdout.flush()

def publish_np_elem(elem):
    gt_pub.publish(to_msg(elem["gt_occ"]))
    known_occ_pub.publish(to_msg(elem["known_occ"]))
    known_free_pub.publish(to_msg(elem['known_free']))
    sys.stdout.write('\033[2K\033[1G')
    print("Category: {}, id: {}, aug: {}".format(elem['shape_category'],
                                                 elem['id'],
                                                 elem['augmentation']), end="")
    sys.stdout.flush()



def publish_options(metadata):
    tso = TextSelectionOptions()

    for i, elem in metadata.enumerate():
        s = elem['id'].numpy() + elem['augmentation'].numpy()
        selection_map[s] = i
        tso.options.append(s)

    options_pub.publish(tso)


def publish_selection(metadata, str_msg):
    translation = 0
    
    ds = metadata.skip(selection_map[str_msg.data]).take(1)
    ds = data_tools.load_voxelgrids(ds)
    # ds = data_tools.simulate_input(ds, 0, 0, 0)
    ds = data_tools.simulate_input(ds, translation, translation, translation)
    # ds = data_tools.simulate_partial_completion(ds)
    # ds = data_tools.simulate_random_partial_completion(ds)

    # Note: there is only one elem in this ds
    elem = next(ds.__iter__())
    publish_elem(elem)

    
    if model is None:
        return
    
        
    elem_expanded = {}
    for k in elem.keys():
        elem_expanded[k] = np.expand_dims(elem[k].numpy(), axis=0)

    inference = model.model(elem_expanded)
    completion_pub.publish(to_msg(inference['predicted_occ'].numpy()))
    completion_free_pub.publish(to_msg(inference['predicted_free'].numpy()))
    mismatch = np.abs(elem['gt_occ'].numpy() - inference['predicted_occ'].numpy())
    mismatch_pub.publish(to_msg(mismatch))

    if SAMPLING:
        global stop_current_sampler
        global sampling_thread
        
        print()
        print("Stopping old worker")
        stop_current_sampler = True
        if sampling_thread is not None:
            sampling_thread.join()
        
        # sampler_worker(elem_expanded)

        sampling_thread = threading.Thread(target=sampler_worker, args=(elem_expanded,))
        sampling_thread.start()
        
        # global sampling_process
        # if sampling_process is not None:
        #     sampling_process.terminate()
        #     print("Terminated Existing Proceed!")
        #     raw_input()

        # sampling_process = Process(target=sampler_worker, args=(elem_expanded,))
        # sampling_process.start()
                                


def sampler_worker(elem):
    global stop_current_sampler
    stop_current_sampler = False

    print()
    sampler = sampling_tools.UnknownSpaceSampler(elem)
    # sampler = sampling_tools.MostConfidentSampler(elem)
    inference = model.model(elem)
        

    finished = False
    # ct = 0
    prev_ct = 0

    while not finished and not stop_current_sampler:
        # a = raw_input()

        try:
            elem, inference = sampler.sample(model, elem, inference)
        except StopIteration:
            finished = True

                
        # ct += 1
        if sampler.ct - prev_ct >= 100 or finished:
            prev_ct = sampler.ct
            ct = 0
            
            publish_np_elem(elem)
            completion_pub.publish(to_msg(inference['predicted_occ'].numpy()))
            completion_free_pub.publish(to_msg(inference['predicted_free'].numpy()))
    print("Sampling complete")
    # IPython.embed()



            


def publish_object_transform():
    # Transform so shapes appear upright in rviz
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = "object"
    q = tf_conversions.transformations.quaternion_from_euler(1.57,0,0)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    rospy.sleep(1)
    br.sendTransform(t)

def load_network():
    global model
    print('Load network? (Y/n)')
    if raw_input().lower() == 'n':
        return
    model = AutoEncoderWrapper()
    

if __name__=="__main__":
    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")

    records = data_tools.load_shapenet_metadata(shuffle=False)
    load_network()

    gt_pub = rospy.Publisher('gt_voxel_grid', OccupancyStamped, queue_size=1)
    known_occ_pub = rospy.Publisher('known_occ_voxel_grid', OccupancyStamped, queue_size=1)
    known_free_pub = rospy.Publisher('known_free_voxel_grid', OccupancyStamped, queue_size=1)
    completion_pub = rospy.Publisher('predicted_occ_voxel_grid', OccupancyStamped, queue_size=1)
    completion_free_pub = rospy.Publisher('predicted_free_voxel_grid', OccupancyStamped, queue_size=1)
    mismatch_pub = rospy.Publisher('mismatch_voxel_grid', OccupancyStamped, queue_size=1)
    options_pub = rospy.Publisher('shapenet_options', TextSelectionOptions, queue_size=1)
    selected_sub = rospy.Subscriber('/shapenet_selection', String,
                                    lambda x: publish_selection(records, x))

    publish_object_transform()
    publish_options(records)

    rospy.spin()

    










    

######  OLD




# def view_single_binvox():
#     gt_pub = rospy.Publisher('gt_voxel_grid', OccupancyStamped, queue_size=1)
#     fp = "/home/bsaund/catkin_ws/src/mps_shape_completion/shape_completion_training/data/ShapeNetCore.v2_augmented/03797390"
#     # fp = os.path.join(fp, "a1d293f5cc20d01ad7f470ee20dce9e0")
#     fp = os.path.join(fp, "214dbcace712e49de195a69ef7c885a4")
#     fp = os.path.join(fp, "models")
#     # fn = "model_normalized.obj_64.binvox"
#     fn = "model_normalized.binvox"
#     # fn = "model_normalized.solid.binvox"

#     fp = os.path.join(fp, fn)
    
#     with open(fp) as f:
#         gt_vox = binvox_rw.read_as_3d_array(f).data

#     print("Publishing single binvox {}".format(fp))
#     gt_pub.publish(to_msg(gt_vox))
#     rospy.sleep(10)


# def publish_test_img():

#     mug_fp = "/home/bsaund/catkin_ws/src/mps_shape_completion/shape_completion_training/data/ShapeNetCore.v2_augmented/03797390/"


#     shapes = os.listdir(mug_fp)
#     shapes = ['214dbcace712e49de195a69ef7c885a4']
    
#     for shape in shapes:
#         shape_fp = os.path.join(mug_fp, shape, "models")
#         print ("Displaying {}".format(shape))

#         gt_fp = os.path.join(shape_fp, "model_normalized.solid.binvox")

#         with open(gt_fp) as f:
#             gt_vox = binvox_rw.read_as_3d_array(f).data
#         gt_pub.publish(to_msg(gt_vox))
#         rospy.sleep(1)

#         augs = [f for f in os.listdir(shape_fp) if f.startswith("model_augmented")]
#         augs.sort()
#         # IPython.embed()
#         for aug in augs:

#             if rospy.is_shutdown():
#                 return
        
#             with open(os.path.join(shape_fp, aug)) as f:
#                 ko_vox = binvox_rw.read_as_3d_array(f).data
    
#             known_occ_pub.publish(to_msg(ko_vox))

#             print("Publishing {}".format(aug))
#             rospy.sleep(.5)




# def publish_shapenet_tfrecords():
#     data = data_tools.load_shapenet([data_tools.shape_map["mug"]], shuffle=False)
#     data = data_tools.simulate_input(data, 5, 5, 5)

#     # print(sum(1 for _ in data))

#     print("")
    
#     for elem in data.batch(1):
#         if rospy.is_shutdown():
#             return
#         publish_elem(elem)
#         rospy.sleep(0.5)


# def publish_completion():

#     print("Loading...")
#     data = data_tools.load_shapenet([data_tools.shape_map["mug"]], shuffle=False)
#     data = data_tools.simulate_input(data, 0, 0, 0)
    
#     model = AutoEncoderWrapper()
#     # model.restore()
#     # model.evaluate(data)
#     print("")

#     for elem in data:

#         if rospy.is_shutdown():
#             return
        
#         dim = elem['gt_occ'].shape[0]


#         elem_expanded = {}
#         for k in elem.keys():
#             elem_expanded[k] = np.expand_dims(elem[k].numpy(), axis=0)

#         inference = model.model(elem_expanded)['predicted_occ'].numpy()
#         publish_elem(elem)

#         completion_pub.publish(to_msg(inference))
#         mismatch = np.abs(elem['gt_occ'].numpy() - inference)
#         mismatch_pub.publish(to_msg(mismatch))
#         # IPython.embed()

#         rospy.sleep(0.5)


# def layer_by_layer():
#     print("Loading...")
#     data = data_tools.load_shapenet([data_tools.shape_map["mug"]])
    
#     model = AutoEncoderWrapper()
#     model.restore()

#     i = 0

#     for elem in data:
#         i += 1
#         if i<8:
#             continue

#         if rospy.is_shutdown():
#             return
#         # IPython.embed()
        
#         print("Publishing")
#         dim = elem['gt_occ'].shape[0]

#         elem_expanded = {}
#         for k in elem.keys():
#             elem_expanded[k] = np.expand_dims(elem[k].numpy(), axis=0)

#         elem_expanded['known_occ'][:,:,32:] = 0.0
#         elem_expanded['known_free'][:,:,32:] = 0.0
        
#         d = 31
#         inference = elem_expanded['known_occ'] * 0.0
#         while d < 64:
#             print("Layer {}".format(d))
#             ko = elem_expanded['known_occ']
#             # IPython.embed()
#             ko[:,:,d] += inference[:,:,d]
#             # IPython.embed()
#             ko=ko.clip(min=0.0, max=1.0)
#             elem_expanded['known_occ'] = ko
#             # IPython.embed()
            

#             inference = model.model.predict(elem_expanded)

#             gt_pub.publish(to_msg(elem['gt'].numpy()))
#             known_occ_pub.publish(to_msg(elem_expanded['known_occ']))
#             known_free_pub.publish(to_msg(elem_expanded['known_free']))
#             completion_pub.publish(to_msg(inference))

#             # if d==31:
#             #     IPython.embed()
            
#             d += 1
#             # rospy.sleep(0.5)
            
#         # IPython.embed()
#         rospy.sleep(2.0)
