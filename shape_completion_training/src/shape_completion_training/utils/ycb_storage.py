from shape_completion_training.model import filepath_tools
from shape_completion_training.model import utils
from shape_completion_training.utils.dataset_storage import load_metadata, _split_train_and_test, write_to_filelist

ycb_load_path = filepath_tools.get_shape_completion_package_path() / "data" / "ycb"
ycb_record_path = ycb_load_path / "tfrecords" / "filepath"


def write_ycb_to_filelist():
    all_files = get_all_ycb_files()

    test_objects = ["002", #master chef can
                    "003", #cracker box
                    "004", #sugar box
                    "005", #tomato soup can
                    "006", #mustard bottle
                    "008", #pudding box
                    "019", #pitcher base
                    "021", #bleach cleanser
                    "022", #windex bottle
                    "035", #power drill
                    "036", #wood block
                    ]

    test_files = []
    for record in all_files:
        if record["id"][0:3] in test_objects:
            test_files.append(record)

    write_to_filelist(utils.sequence_of_dicts_to_dict_of_sequences(all_files),
                      ycb_record_path / "train_filepaths.pkl")
    write_to_filelist(utils.sequence_of_dicts_to_dict_of_sequences(test_files),
                      ycb_record_path / "test_filepaths.pkl")


def get_all_ycb_files():
    records = []
    obj_fps = [fp for fp in ycb_load_path.iterdir() if fp.stem.startswith("0")]
    for obj_fp in sorted(obj_fps):

        print("{}".format(obj_fp.name))
        all_augmentation = [f for f in (obj_fp / "google_16k").iterdir()
                            if f.name.startswith("model_augmented")
                            if f.name.endswith(".pkl.gzip")]
        for f in sorted(all_augmentation):
            base = f.parent / f.stem
            records.append(load_metadata(base, compression="gzip"))
    return records
