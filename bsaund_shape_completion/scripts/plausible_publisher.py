from __future__ import print_function
import rospy
import shape_completion_training.model.observation_model
from shape_completion_training.utils import data_tools
from bsaund_shape_completion.voxelgrid_publisher import VoxelgridPublisher
from bsaund_shape_completion.shape_selection import send_display_names_from_metadata
# from shape_completion_training.voxelgrid import fit
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.model import plausiblility
from shape_completion_training.model import default_params

DATASET = "ycb"

default_dataset_params = default_params.get_default_params()
default_dataset_params.update({
    'apply_slit_occlusion': True,
})

dataset_params = default_dataset_params


def publish_selection(metadata, ind, str_msg):
    ds = metadata.skip(ind).take(1)
    ds = data_tools.load_voxelgrids(ds)
    ds = data_tools.preprocess_test_dataset(ds, dataset_params)
    # ds = data_tools.simulate_input(ds, 0, 0, 0, sim_input_fn=data_tools.simulate_2_5D_input)
    # ds = data_tools.apply_fixed_slit_occlusion(ds, 35, 6)
    elem = next(ds.__iter__())
    elem['gt_occ'] = data_tools.shift_voxelgrid(elem['gt_occ'], 1, 0, 0, 0, 1, 1, 1)
    VG_PUB.publish_elem(elem)



    fit_2_5D_view(metadata, elem)


def fit_2_5D_view(metadata, reference):
    sn = data_tools.get_addressible_dataset(use_train=False, dataset_name=DATASET)
    print("Loading plausibilities")

    valid_fits = plausiblility.get_plausibilities_for(data_tools.get_unique_name(reference), dataset_name=DATASET)
    print("plausibilities loaded")

    for elem_name, T, p, oob in valid_fits:
        elem = sn.get(elem_name)

        fitted = conversions.transform_voxelgrid(elem['gt_occ'], T, scale=0.01)
        VG_PUB.publish("plausible", fitted)
        p = shape_completion_training.model.observation_model.observation_likelihood_geometric_mean(reference['gt_occ'],
                                                                                                    fitted,
                                                                                                    std_dev_in_voxels=2)
        print("Best fit for {}: p={}".format(data_tools.get_unique_name(elem), p))
        rospy.sleep(0.1)


if __name__ == "__main__":
    rospy.init_node('plausible_shape_publisher')
    rospy.loginfo("Data Publisher")

    train_records, test_records = data_tools.load_dataset(dataset_name=DATASET, metadata_only=True,
                                                          shuffle=False)

    VG_PUB = VoxelgridPublisher()

    selection_sub = send_display_names_from_metadata(test_records, publish_selection)

    rospy.spin()
