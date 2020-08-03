from shape_completion_training.model import utils
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils import data_tools
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher

from visualization_msgs.msg import Marker
import rospy
from time import time

model_name_map = {"NormalizingAE/July_02_15-15-06_ede2472d34": "PSSNet (ours)",
                  "VAE_GAN/July_20_23-46-36_8849b5bd57": "VAE-GAN",
                  "3D_rec_gan/July_20_19-36-48_7ed486bdf5": "3D-rec-GAN",
                  "VAE/July_07_12-09-24_7f65111254": "VAE",
                  "GT_and_input": "2.5D"}

models = [k for k in model_name_map]


def publish_name(publisher, name):
    m = Marker()
    m.type = m.TEXT_VIEW_FACING
    m.scale.z = 0.1
    m.pose.position.z = 0.7
    m.pose.position.x = .3
    m.pose.position.y = -.3

    m.text = model_name_map[model_name]
    m.header.frame_id = "world"
    m.color.a = 1.0
    publisher.publish(m)

def publish_display_count(publisher, count):
    m = Marker()
    m.type = m.TEXT_VIEW_FACING
    m.scale.z = 0.02
    m.pose.position.z = 0.3
    m.pose.position.x = .6
    m.pose.position.y = -.6

    m.text = str(count)
    m.header.frame_id = "world"
    m.color.a = 1.0
    publisher.publish(m)


if __name__ == "__main__":
    rospy.init_node('shapenet_movie_maker')
    VG_PUB = VoxelgridPublisher()
    name_pub = rospy.Publisher("model_name", Marker, queue_size=1)
    count_pub = rospy.Publisher("display_count", Marker, queue_size=1)

    for model_name in models:

        if model_name is not "GT_and_input":
            model_runner = ModelRunner(training=False, trial_path=model_name)
        publish_name(name_pub, model_name)
        dataset_params = model_runner.params
        train_records, test_records = data_tools.load_dataset(dataset_name=dataset_params['dataset'],
                                                              metadata_only=True, shuffle=False)
        ds = data_tools.load_voxelgrids(test_records)
        ds = data_tools.preprocess_test_dataset(ds, dataset_params)

        for elem_num, elem in ds.enumerate():
            publish_display_count(count_pub, elem_num.numpy())
            if rospy.is_shutdown():
                exit(0)

            elem = utils.add_batch_to_dict(elem)
            for i in range(5):
                inference = model_runner.model(elem)
                VG_PUB.publish_elem(elem)

                if model_name is "GT_and_input":
                    inference['predicted_occ'] = inference['predicted_occ'] * 0

                VG_PUB.publish_inference(inference)
