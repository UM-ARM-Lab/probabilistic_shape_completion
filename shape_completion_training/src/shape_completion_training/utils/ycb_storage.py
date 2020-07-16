from shape_completion_training.model import filepath_tools
from shape_completion_training.model import utils
from shape_completion_training.utils.dataset_storage import load_metadata, _split_train_and_test, write_to_filelist


ycb_load_path = filepath_tools.get_shape_completion_package_path() / "data" / "ycb"
ycb_record_path = ycb_load_path / "tfrecords" / "filepath"


def write_ycb_to_filelist(test_ratio):
    all_files = get_all_ycb_files()
    train_files, test_files = _split_train_and_test(all_files, test_ratio)

    write_to_filelist(utils.sequence_of_dicts_to_dict_of_sequences(train_files),
                      ycb_record_path / "train_filepaths.pkl")
    write_to_filelist(utils.sequence_of_dicts_to_dict_of_sequences(test_files),
                      ycb_record_path / "test_filepaths.pkl")

def get_all_ycb_files():
    records = []
    for obj_fp in ycb_load_path.iterdir():
        print("{}".format(obj_fp.namev))
        all_augmentation = [f for f in (obj_fp / "google_16k").iterdir()
                            if f.name.startswith("model_augmented")
                            if f.name.endswith(".pkl.gzip")]
        for f in sorted(all_augmentation):
            base = f.parent / f.stem
            records.append(load_metadata(base, compression="gzip"))
    return records