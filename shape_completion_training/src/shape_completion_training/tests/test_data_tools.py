from unittest import TestCase
import shape_completion_training.model.data_tools as data_tools


class TestDatasetLoading(TestCase):
    def test_dataset_exists_and_can_be_loaded(self):
        data_shapenet = data_tools.load_shapenet([data_tools.shape_map["mug"]])


