import json
import pathlib
import unittest

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

    def test_get_trial_directory_load_existing(self):
        expected_trial_path = pathlib.Path('.testing/new_trial/subdir')
        params_filename = expected_trial_path / 'params.json'
        expected_params = filepath_tools.get_default_params()
        expected_params['a'] = 2
        with params_filename.open("w") as params_file:
            json.dump(expected_params, params_file)
        expected_trial_path.mkdir(parents=True, exist_ok=True)

        trial_path, loaded_params = filepath_tools.create_or_load_trial(trial_path=expected_trial_path,
                                                                        write_summary=False)
        self.assertEqual(expected_trial_path, trial_path)
        self.assertEqual(loaded_params, expected_params)


if __name__ == '__main__':
    unittest.main()
