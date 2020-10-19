import json
import pathlib
import unittest

import shape_completion_training.model.default_params
from shape_completion_training.model import filepath_tools


class FilePathToolsTest(unittest.TestCase):
    default_params_filename = None

    def test_get_trial_directory_new_train(self):
        base_dir = pathlib.Path('.testing')
        group_name = 'new_trial'
        trial_directory, params = filepath_tools.create_or_load_trial(group_name=group_name,
                                                                      params={},
                                                                      trials_directory=base_dir,
                                                                      write_summary=False)
        trial_directory = trial_directory
        self.assertTrue(trial_directory.exists())
        self.assertEqual(trial_directory.parent, pathlib.Path(base_dir) / group_name)
        filepath_tools.rm_tree(trial_directory)

    def test_get_trial_directory_load_nonexistant(self):
        trial_path = pathlib.Path('.testing/new_trial/subdir_non_existing')
        self.assertRaises(ValueError,
                          filepath_tools.create_or_load_trial,
                          trial_path=trial_path,
                          write_summary=False)


if __name__ == '__main__':
    unittest.main()
