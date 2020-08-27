#! /usr/bin/env python

import numpy as np
import math
import random
import tensorflow as tf
import rospy
from rviz_voxelgrid_visuals import conversions
from shape_completion_training.voxelgrid.conversions import pointcloud_to_voxelgrid
from shape_completion_training.model.model_runner import ModelRunner
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# filedir = "../data/my_ds"

# random center
# random orientation
# random length
# random view

def generate_cylinder(center, orient, height, radius):
	cylinder = np.zeros([64, 64, 64, 1])
	for x in range(64):
		for y in range(64):
			for z in range(64):
				coor = np.array([x, y, z]).astype(float)
				diff = coor - center
				proj_height = abs(np.dot(orient, diff))
				proj_radius = math.sqrt(np.dot(diff, diff) - proj_height*proj_height)
				if proj_height <= height and proj_radius <= radius:
					cylinder[x, y, z, 0] = 1
	return cylinder

def generate_partial_cylinder(cylinder):
	blocked = np.zeros([64, 64])
	partial = np.zeros([64, 64, 64, 1])
	for x in range(64):
		for y in range(64):
			for z in range(64):
				if cylinder[x, y, z, 0] == 1 and blocked[y, z] == 0:
					partial[x, y, z, 0] = 1
					blocked[y, z] = 1
				elif cylinder[x, y, z, 0] == 1 and blocked[y, z] == 1:
					partial[x, y, z, 0] = 1
					blocked[y, z] = 2
	return partial

def rot_zyx(voxel, z_rot, y_rot, x_rot):
	rot_voxel = np.zeros(voxel.shape)
	for x in range(voxel.shape[0]):
		for y in range(voxel.shape[1]):
			for z in range(voxel.shape[2]):
				if voxel[x, y, z, 0] == 1:
					diff_x = float(x) - 32.0
					diff_y = float(y) - 32.0
					diff_z = float(z) - 32.0

					x_1 = diff_x*math.cos(z_rot) - diff_y*math.sin(z_rot)
					y_1 = diff_x*math.sin(z_rot) + diff_y*math.cos(z_rot)
					z_1 = diff_z

					x_2 = x_1*math.cos(y_rot) + z_1*math.sin(y_rot)
					y_2 = y_1
					z_2 = -x_1*math.sin(y_rot) + z_1*math.cos(y_rot)

					x_3 = x_2 + 32.0
					y_3 = y_2*math.cos(x_rot) - z_2*math.sin(x_rot) + 32.0
					z_3 = y_2*math.sin(x_rot) + z_2*math.cos(x_rot) + 32.0

					rot_voxel[int(round(x_3)), int(round(y_3)), int(round(z_3)), 0] = 1
	return rot_voxel

def random_rot(voxel):
	random.seed()
	z_rot = random.random()*2*math.pi
	y_rot = random.random()*2*math.pi
	x_rot = random.random()*2*math.pi
	ran_voxel = rot_zyx(voxel, z_rot, y_rot, x_rot)
	return ran_voxel

def main():
	random.seed()
	known_occ_list = []
	gt_occ_list = []
	known_free_list = []
	gt_free_list = []
	rospy.init_node("cdcpd_shape_completion")
	pub_incomp = rospy.Publisher('incomp', VoxelgridStamped, queue_size=1)
	pub_comp = rospy.Publisher('comp', VoxelgridStamped, queue_size=1)
	scale = 1.0
	origin = (0.0, 0.0, 0.0)
	alphas = [0.0, math.pi/4.0, math.pi/2.0, math.pi*3.0/4.0]
	betas = [0.0, math.pi/4.0, math.pi/2.0, math.pi*3.0/4.0]
	rot_zyx_angle = [[0.0, 0.0, 0.0],
					 [math.pi/2.0, 0.0, 0.0],
					 [math.pi, 0.0, 0.0],
					 [math.pi*3.0/2.0, 0.0, 0.0],
					 [0.0, math.pi/2.0, 0.0],
					 [0.0, 3.0*math.pi/2.0, 0.0]]

	for num_cylinder in range(10):
		x = random.random()*10.0 + 27.0
		y = random.random()*10.0 + 27.0
		z = random.random()*10.0 + 27.0
		center = np.array([x, y, z])
		height = random.random()*10.0 + 10.0
		radius = random.random()*4.0+4.0
		for alpha_ind in range(len(alphas)):
			alpha = alphas[alpha_ind]
			if alpha_ind != 3:
				for beta_ind in range(len(betas)):
					beta = betas[beta_ind]
					orient = np.array([math.cos(alpha)*math.cos(beta), math.cos(alpha)*math.sin(beta), math.sin(alpha)])
					gt_occ = generate_cylinder(center, orient, height, radius)
					known_occ = generate_partial_cylinder(gt_occ)
					zero_space = np.zeros(known_occ.shape)
					for rot_angle in rot_zyx_angle:
						gt_occ_rot = rot_zyx(gt_occ, rot_angle[0], rot_angle[1], rot_angle[2])
						known_occ_rot = rot_zyx(known_occ, rot_angle[0], rot_angle[1], rot_angle[2])
						pub_incomp.publish(conversions.vox_to_voxelgrid_stamped(known_occ_rot[:, :, :, 0], # Numpy or Tensorflow
																	scale=scale, # Each voxel is a 1cm cube
																	frame_id='world', # In frame "world", same as rviz fixed frame
																	origin=origin)) # Bottom left corner
						pub_comp.publish(conversions.vox_to_voxelgrid_stamped(gt_occ_rot[:, :, :, 0], # Numpy or Tensorflow
																  scale=scale, # Each voxel is a 1cm cube
																  frame_id='world', # In frame "world", same as rviz fixed frame
																  origin=origin)) # Bottom left corner
			
						known_occ_list.append(known_occ_rot.astype('float32'))
						gt_occ_list.append(gt_occ_rot.astype('float32'))
						known_free_list.append(zero_space.astype('float32'))
						gt_free_list.append((1.0 - gt_occ_rot).astype('float32'))
			else:
				beta = 0.0
				orient = np.array([math.cos(alpha)*math.cos(beta), math.cos(alpha)*math.sin(beta), math.sin(alpha)])
				gt_occ = generate_cylinder(center, orient, height, radius)
				known_occ = generate_partial_cylinder(gt_occ)
				zero_space = np.zeros(known_occ.shape)
				for rot_angle in rot_zyx_angle:
					gt_occ = rot_zyx(gt_occ, rot_angle[0], rot_angle[1], rot_angle[2])
					known_occ = rot_zyx(known_occ, rot_angle[0], rot_angle[1], rot_angle[2])
					pub_incomp.publish(conversions.vox_to_voxelgrid_stamped(known_occ[:, :, :, 0], # Numpy or Tensorflow
																	scale=scale, # Each voxel is a 1cm cube
																	frame_id='world', # In frame "world", same as rviz fixed frame
																	origin=origin)) # Bottom left corner
					pub_comp.publish(conversions.vox_to_voxelgrid_stamped(gt_occ[:, :, :, 0], # Numpy or Tensorflow
																  scale=scale, # Each voxel is a 1cm cube
																  frame_id='world', # In frame "world", same as rviz fixed frame
																  origin=origin)) # Bottom left corner
			
					known_occ_list.append(known_occ.astype('float32'))
					gt_occ_list.append(gt_occ.astype('float32'))
					known_free_list.append(zero_space.astype('float32'))
					gt_free_list.append((1.0 - gt_occ).astype('float32'))

	ds = tf.data.Dataset.from_tensor_slices({'known_occ': known_occ_list,
                                             'gt_occ': gt_occ_list,
                                             'known_free': known_free_list,
                                             'gt_free': gt_free_list})
	print(ds)
	params = {
        'batch_size': 4,
        'dataset': 'shapenet',
        'network': '3D_rec_gan',
        "learning_rate": 0.0001,
        "gan_learning_rate": 0.00005,
        "num_latent_layers": 2000,
        "is_u_connected": True,
        'dataset': 'ycb',
        'apply_slit_occlusion': True,
        'translation_pixel_range_x': 15,
        'translation_pixel_range_y': 10,
        'translation_pixel_range_z': 10,
	}
	mr = ModelRunner(training=False, params=params, group_name='3D_rec_gan')
	mr.train(ds)

if __name__ == "__main__":
    main()