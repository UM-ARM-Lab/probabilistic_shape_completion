from shape_completion_training.model import data_tools

if __name__ == "__main__":
    shapenet = data_tools.load_shapenet([data_tools.shape_map["mug"]])
    elem = next(shapenet.__iter__())
    print(elem)
