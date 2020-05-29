from shape_completion_training.model import data_tools


def process(ds):
    names = set()
    num = 0
    for elem in ds:
        names.add(elem['id'].numpy())
        num += 1
    return {"names": names, "num_elements": num}


def report(ds_name, info):
    print("{} has:".format(ds_name))
    for k, v in info.items():
        print("{}: {}".format(k, v))


if __name__ == "__main__":
    train_ds, test_ds = data_tools.load_shapenet_metadata()
    print(report("test dataset", process(test_ds)))

