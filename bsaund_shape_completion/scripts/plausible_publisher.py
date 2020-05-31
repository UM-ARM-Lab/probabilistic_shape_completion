from __future__ import print_function
import rospy
from shape_completion_training.model import data_tools
from bsaund_shape_completion.voxelgrid_publisher import VoxelgridPublisher
from bsaund_shape_completion.shape_selection import send_display_names_from_metadata
# from shape_completion_training.voxelgrid import fit
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.model import plausiblility
from shape_completion_training.model import model_evaluator


def publish_selection(metadata, ind, str_msg):
    if ind == 0:
        print("Skipping first display")
        return

    ds = metadata.skip(ind).take(1)
    ds = data_tools.load_voxelgrids(ds)
    ds = data_tools.simulate_input(ds, 0, 0, 0)
    elem = next(ds.__iter__())
    VG_PUB.publish_elem(elem)
    fit_2_5D_view(metadata, elem)


def fit_2_5D_view(metadata, reference):
    # ds = metadata
    # ds = data_tools.load_voxelgrids(ds)
    # ds = data_tools.simulate_input(ds, 0, 0, 0)
    # sn = data_tools.AddressableShapenet(use_train=False)
    sn = data_tools.get_addressible_shapenet(use_train=False)
    print("Loading plausibilities")
    best_fits = plausiblility.get_fits_for(data_tools.get_unique_name(reference))
    print("plausibilities loaded")

    for elem_name, T, p in best_fits:
        elem = sn.get(elem_name)
        # T = best_fits[data_tools.get_unique_name(reference)][data_tools.get_unique_name(elem)]
        # T = fit.icp_transform(elem["known_occ"], reference, scale=0.01)
        fitted = conversions.transform_voxelgrid(elem['gt_occ'], T, scale=0.01)
        VG_PUB.publish("sampled_occ", fitted)
        p = model_evaluator.observation_likelihood_geometric_mean(reference['gt_occ'], fitted, std_dev_in_voxels=2)
        print("Best fit for {}: p={}".format(data_tools.get_unique_name(elem), p))
        rospy.sleep(0.1)


if __name__ == "__main__":
    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")

    train_records, test_records = data_tools.load_shapenet_metadata(shuffle=False)

    VG_PUB = VoxelgridPublisher()

    selection_sub = send_display_names_from_metadata(test_records, publish_selection)

    rospy.spin()
