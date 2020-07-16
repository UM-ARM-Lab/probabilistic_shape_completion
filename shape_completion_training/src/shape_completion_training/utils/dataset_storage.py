import bz2
import gzip
import pickle
from os.path import join

import numpy as np
import pathlib
import tensorflow as tf

from shape_completion_training import binvox_rw
from shape_completion_training.model import filepath_tools
from shape_completion_training.voxelgrid import bounding_box

package_path = filepath_tools.get_shape_completion_package_path()


def _load_compressed(filepath, compression):
    """
    @param filepath: full or relative (from shape_completion_training)
    @param compression:
    @return:
    """
    if not filepath.is_absolute():
        filepath = package_path / filepath
    if compression == "bz2":
        with bz2.BZ2File(filepath.with_suffix(".pkl.bz2").as_posix()) as f:
            return pickle.load(f)
    if compression == "gzip":
        with gzip.open(filepath.with_suffix(".pkl.gzip").as_posix()) as f:
            return pickle.load(f)

    with filepath.with_suffix(".pkl").open() as f:
        return pickle.load(f)


def save_gt_voxels(filepath, gt, compression="gzip"):
    """
    @param filepath: pathlib.Path full file path of save location.
    This needs to be in the proper place in the shapenet folder structure, as parameters are inferred from
    the folder name. The extension is replaced with .pkl
    @param gt: The ground truth voxelgrid
    @param compression: "gzip", "bz2", or None
    @return:
    """
    filepath = filepath.with_suffix(".pkl")
    shape = tf.TensorShape(gt.shape)
    packed = np.packbits(gt.flatten().astype(bool))
    parts = filepath.parts
    scale = 0.01
    augmentation = filepath.stem[len("model_augmented_"):]
    x_angle = int(augmentation.split("_")[-3])
    y_angle = int(augmentation.split("_")[-2])
    z_angle = int(augmentation.split("_")[-1])
    bb = bounding_box.get_bounding_box_for_elem(gt, x_angle, y_angle, z_angle,
                                                scale=scale, degrees=True)
    data = {"gt_occ_packed": packed, "shape": tf.TensorShape(shape), "augmentation": augmentation,
            "filepath": filepath.relative_to(package_path).as_posix(),
            "category": parts[-4], "id": parts[-3],
            # "angle": angle,
            "x_angle": x_angle, "y_angle": y_angle, "z_angle": z_angle,
            "bounding_box": bb,
            "scale": scale
            }

    if compression == "bz2":
        with bz2.BZ2File(filepath.with_suffix(".pkl.bz2").as_posix(), "w") as f:
            pickle.dump(data, f)
    elif compression == "gzip":
        with gzip.open(filepath.with_suffix(".pkl.gzip").as_posix(), "w") as f:
            pickle.dump(data, f)
    else:
        with open(filepath.as_posix(), "w") as f:
            pickle.dump(data, f)


def load_data_with_gt(filepath, compression="gzip"):
    """
    @param filepath: pathlib.Path filepath to record
    @return: dictionary with gt_voxels and metadata
    """
    loaded = _load_compressed(filepath, compression)
    loaded["gt_occ"] = np.reshape(np.unpackbits(loaded['gt_occ_packed']), loaded['shape']).astype(np.float32)
    loaded.pop("gt_occ_packed")
    return loaded


def load_gt_only(filepath, compression="gzip"):
    loaded = _load_compressed(pathlib.Path(filepath), compression)
    return np.reshape(np.unpackbits(loaded['gt_occ_packed']), loaded['shape']).astype(np.float32)


def load_metadata(filepath, compression="gzip"):
    """
    @param filepath: filepath: pathlib.Path filepath to record
    @return: dictionary with just metadata
    """
    loaded = _load_compressed(filepath, compression)
    loaded.pop("gt_occ_packed")
    return loaded


def load_gt_voxels_from_binvox(filepath, augmentation):
    """
    Loads ground truth voxels into a np.array

    filepath: string filepath to the "models" folder for this shape
    augmentation: string identifying the augmentation
    """
    binvox_wire_fp = join(filepath, 'model_augmented_' + augmentation + '.wire.binvox')
    with open(binvox_wire_fp) as f:
        wire_vox = binvox_rw.read_as_3d_array(f).data

    binvox_mesh_fp = join(filepath, 'model_augmented_' + augmentation + '.mesh.binvox')
    with open(binvox_mesh_fp) as f:
        mesh_vox = binvox_rw.read_as_3d_array(f).data

    # cuda_binvox_fp = join(filepath, 'model_augmented_' + augmentation + '.obj_64.binvox')
    # with open(cuda_binvox_fp) as f:
    #     cuda_gt_vox = binvox_rw.read_as_3d_array(f).data

    gt = wire_vox * 1.0 + mesh_vox * 1.0
    gt = np.clip(gt, 0, 1)
    gt = np.array(gt, dtype=np.float32)
    gt = np.expand_dims(gt, axis=4)
    return gt


def _split_train_and_test(shapenet_records, test_ratio):
    train_ids = []
    test_ids = []
    train_records = []
    test_records = []
    np.random.seed(42)
    for record in shapenet_records:
        if record["id"] not in train_ids and record["id"] not in test_ids:
            if np.random.random() < test_ratio:
                test_ids.append(record["id"])
            else:
                train_ids.append(record["id"])

        if record["id"] in train_ids:
            train_records.append(record)
        else:
            test_records.append(record)

    return train_records, test_records


def write_to_filelist(dataset, record_file):
    # def _bytes_feature(value):
    #     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    #
    # with tf.io.TFRecordWriter(record_file.as_posix()) as writer:
    #     for elem in dataset:
    #         feature = {k: _bytes_feature(elem[k].numpy()) for k in elem}
    #         features = tf.train.Features(feature=feature)
    #         example = tf.train.Example(features=features)
    #         writer.write(example.SerializeToString())
    with open(record_file.as_posix(), "w") as f:
        pickle.dump(dataset, f)

    with open(record_file.as_posix()) as f:
        ds = pickle.load(f)