#! /usr/bin/env python
import rospy
from shape_completion_training.model import model_evaluator
from bsaund_shape_completion.voxelgrid_publisher import VoxelgridPublisher
from shape_completion_training.utils import data_tools
from shape_completion_training.model import utils
from shape_completion_training.model.model_runner import ModelRunner
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from shape_completion_training.voxelgrid.metrics import chamfer_distance
from shape_completion_training.model import filepath_tools
from scipy.optimize import linear_sum_assignment

save_folder = filepath_tools.get_shape_completion_package_path() / "results"

# data_names = ["model", "shape", "best_sample_gt", ""

# Shapenet
# sn = data_tools.get_addressible_dataset(use_train=False)
# model_name_map = {"VAE/July_07_12-09-24_7f65111254": "VAE",
#                   "3D_rec_gan/July_20_19-36-48_7ed486bdf5": " 3DrecGAN++",
#                   "VAE_GAN/July_20_23-46-36_8849b5bd57": "VAE_GAN",
#                   # "VAE/VAE_trial_1": "Baseline VAE",
#                   # "Augmented_VAE/May_21_20-00-00_0000000000": "Our method",
#                   "NormalizingAE/July_02_15-15-06_ede2472d34": "Ours",
#                   }

# YCB
sn = data_tools.get_addressible_dataset(dataset_name="ycb")
model_name_map = {"NormalizingAE_YCB/July_24_11-21-46_f2aea4d768": "Ours",
                  "VAE_YCB/July_24_11-21-49_f2aea4d768": "VAE",
                  "VAE_GAN_YCB/July_25_22-50-44_0f55a0f6b3": "VAE_GAN",
                  "3D_rec_gan_YCB/July_25_22-51-08_0f55a0f6b3": "3DrecGAN++"
}

def save_numerics(df):
    averages = df.groupby('model').mean()
    averages.to_csv((save_folder / "all_averages.csv").as_posix())
    ambiguous = df[df['angle'] > 249]
    ambiguous = ambiguous[ambiguous['angle'] < 289]
    ambiguous_averages = ambiguous.groupby('model').mean()
    ambiguous_averages.to_csv((save_folder / "ambiguous_averages.csv").as_posix())


def add_optimal_assignment(data, shape_name, model_name, particle_distances):
    d = np.array(particle_distances)
    d = d[:, ~np.any(np.isnan(d), axis=0)]
    angle = int(sn.get_metadata(shape_name)['augmentation'].numpy()[-3:])
    try:
        opt_assignment = linear_sum_assignment(d)
    except:
        pass
    opt_match_distance = d[opt_assignment]
    for match_dist in opt_match_distance:
        data["model"].append(model_name_map[model_name])
        data["shape"].append(shape_name)
        data["Chamfer Distance from each Plausible to closest Sample"].append(None)
        data["angle"].append(angle)
        data["Chamfer Distance from each Sample to closest Plausible"].append(None)
        data["accuracy"].append(None)
        data["hausdorff"].append(None)
        data["assignment"].append(match_dist)



def add_coverage(data, shape_name, model_name, particle_distances):
    angle = int(sn.get_metadata(shape_name)['augmentation'].numpy()[-3:])
    for closest_particle in list(np.min(particle_distances, axis=1)):
        data["model"].append(model_name_map[model_name])
        data["shape"].append(shape_name)
        data["Chamfer Distance from each Plausible to closest Sample"].append(closest_particle)
        data["angle"].append(angle)
        data["Chamfer Distance from each Sample to closest Plausible"].append(None)
        data["accuracy"].append(None)
        data["hausdorff"].append(None)
        data["assignment"].append(None)


def add_plausibility(data, shape_name, model_name, particle_distances):
    angle = int(sn.get_metadata(shape_name)['augmentation'].numpy()[-3:])
    for closest_particle in list(np.min(particle_distances, axis=0)):
        data["model"].append(model_name_map[model_name])
        data["shape"].append(shape_name)
        data["Chamfer Distance from each Sample to closest Plausible"].append(closest_particle)
        data["angle"].append(angle)
        data["Chamfer Distance from each Plausible to closest Sample"].append(None)
        data["accuracy"].append(None)
        data["hausdorff"].append(None)
        data["assignment"].append(None)



def add_accuracy(data, shape_name, model_name, distance_to_gt):
    angle = int(sn.get_metadata(shape_name)['augmentation'].numpy()[-3:])
    data["model"].append(model_name_map[model_name])
    data["shape"].append(shape_name)
    data["Chamfer Distance from each Plausible to closest Sample"].append(None)
    data["angle"].append(angle)
    data["Chamfer Distance from each Sample to closest Plausible"].append(None)
    data["accuracy"].append(distance_to_gt)
    data["hausdorff"].append(None)
    data["assignment"].append(None)


def add_hausdorff(data, shape_name, model_name, particle_distances):
    angle = int(sn.get_metadata(shape_name)['augmentation'].numpy()[-3:])
    # for closest_particle in np.max(np.min(particle_distances, axis=0)):
    hausdorff_distance = max(np.max(np.min(particle_distances, axis=0)), np.max(np.min(particle_distances, axis=1)))
    data["model"].append(model_name_map[model_name])
    data["shape"].append(shape_name)
    data["Chamfer Distance from each Sample to closest Plausible"].append(None)
    data["angle"].append(angle)
    data["Chamfer Distance from each Plausible to closest Sample"].append(None)
    data["accuracy"].append(None)
    data["hausdorff"].append(hausdorff_distance)
    data["assignment"].append(None)


def process_evaluation_into_dataframe(evaluation):
    data = {name: [] for name in ["model", "shape",
                                  "Chamfer Distance from each Plausible to closest Sample",
                                  "Chamfer Distance from each Sample to closest Plausible",
                                  "angle", "accuracy", "hausdorff", "assignment"]}
    for model_name, model_evaluation in evaluation.items():
        print("Processing data for {}".format(model_name))
        for shape_name, shape_evaluation in model_evaluation.items():
            # if not 250 < angle < 290:
            #     continue
            d = shape_evaluation['particle_distances']
            if np.array(d).shape[0] == 0:
                print("No particle distances for {}".format(shape_name))
                continue
            add_coverage(data, shape_name, model_name, d)
            add_plausibility(data, shape_name, model_name, d)
            add_accuracy(data, shape_name, model_name, shape_evaluation['best_particle_chamfer'])
            add_hausdorff(data, shape_name, model_name, d)
            add_optimal_assignment(data, shape_name, model_name, d)

    return pd.DataFrame(data, columns=data.keys())


def display_histogram(df):
    sns.set(style="darkgrid")

    sns.lineplot(x="shape", y="Chamfer Distance from each Plausible to closest Sample", data=df, hue="model")
    # plt.show()
    plt.savefig((save_folder / "Coverage.png").as_posix())
    plt.clf()

    sns.lineplot(x="shape", y="Chamfer Distance from each Sample to closest Plausible", data=df, hue="model")
    # plt.show()
    plt.savefig((save_folder / "Plausibility.png").as_posix())
    plt.clf()

    sns.lineplot(x="shape", y="accuracy", data=df, hue="model")
    # plt.show()
    plt.savefig((save_folder / "Accuracy.png").as_posix())
    plt.clf()

    sns.lineplot(x="shape", y="hausdorff", data=df, hue="model")
    plt.savefig((save_folder / "Hausdorff.png").as_posix())
    plt.clf()

    sns.lineplot(x="shape", y="assignment", data=df, hue="model")
    plt.savefig((save_folder / "assignment.png").as_posix())
    plt.clf()

    for unique_shape in set([s[0:10] for s in list(df['shape'])]):
        print(unique_shape)
        shape_df = df[df['shape'].str.startswith(unique_shape)]
        fig = sns.lineplot(x="angle", y="Chamfer Distance from each Plausible to closest Sample", data=shape_df,
                           hue="model")
        # plt.show()
        fig.set_ylim([0, 0.007])
        fig.get_legend().remove()
        plt.savefig((save_folder / (unique_shape + ".png")).as_posix())
        plt.clf()


def display_voxelgrids(evaluation):
    # model_name = evaluation.keys()[1]
    model_name = "Augmented_VAE/May_21_20-00-00_0000000000"
    print("Showing results for {}".format(model_name))
    model = ModelRunner(training=False, trial_path=model_name).model
    # shape_name = evaluation[model_name].keys()[0]
    # shape_evaluation = evaluation[model_name][shape_name]
    for shape_name, shape_evaluation in evaluation[model_name].items():
        if not shape_name.startswith('9737'):
            continue
        angle = int(sn.get_metadata(shape_name)['augmentation'].numpy()[-3:])
        # if not 250 < angle < 290:
        if not 329 < angle < 331:
            continue
        # display_evaluation_for_shape(model, shape_name, shape_evaluation)
        print("Getting plausibilities")
        plausibles = model_evaluator.get_plausibles(shape_name)

        elem = sn.get(shape_name)
        # e = {k: tf.expand_dims(v, axis=0) for k, v in elem.items()}
        print("Sampling particles for {}".format(shape_name))
        particles = model_evaluator.sample_particles(model, utils.add_batch_to_dict(elem),
                                                     num_particles=len(shape_evaluation['particle_distances'][0]))
        VG_PUB.publish_elem(elem)

        best_particles = np.argmin(shape_evaluation['particle_distances'], axis=1)
        for i, plausible in enumerate(plausibles):
            VG_PUB.publish("plausible", plausible)
            VG_PUB.publish("predicted_occ", particles[best_particles[i]])
            print("Distance is {}".format(np.min(shape_evaluation['particle_distances'], axis=1)[i]))
            print("Recomputed  {}".format(chamfer_distance(plausible, particles[best_particles[i]],
                                                           scale=0.01, downsample=4)))
            rospy.sleep(1)


def display_evaluation_for_shape(model, shape_name, shape_evaluation):
    plausibles = model_evaluator.get_plausibles(shape_name)

    elem = sn.get(shape_name)
    # e = {k: tf.expand_dims(v, axis=0) for k, v in elem.items()}
    print("Sampling particles for {}".format(shape_name))
    particles = model_evaluator.sample_particles(model, utils.add_batch_to_dict(elem),
                                                 num_particles=len(shape_evaluation['particle_distances'][0]))
    VG_PUB.publish_elem(elem)

    best_particles = np.argmin(shape_evaluation['particle_distances'], axis=1)
    for i, plausible in enumerate(plausibles):
        VG_PUB.publish("plausible", plausible)
        VG_PUB.publish("predicted_occ", particles[best_particles[i]])
        rospy.sleep(1)


def display_coverage(particles, plausibles):
    pass


if __name__ == "__main__":
    rospy.init_node('evaluation_display')
    VG_PUB = VoxelgridPublisher()

    full_evaluation = model_evaluator.load_evaluations(model_name_map.keys())
    print("Loading addressable shapenet")

    # display_voxelgrids(full_evaluation)
    df = process_evaluation_into_dataframe(full_evaluation)
    display_histogram(df)
    save_numerics(df)
