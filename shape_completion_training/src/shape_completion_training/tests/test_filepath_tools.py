import json
import pathlib
import unittest

from shape_completion_training.model import filepath_tools


class FilePathToolsTest(unittest.TestCase):
    default_params_filename = None

    def test_get_trial_directory_new_train(self):
        base_dir = pathlib.Path('.testing')
        unique_nickname = 'new_trial'
        trial_directory, params, training = filepath_tools.create_or_load_trial(trial_name=unique_nickname,
                                                                                params={},
                                                                                write_summary=False,
                                                                                trials_directory=base_dir)
        trial_directory = trial_directory
        self.assertTrue(trial_directory.exists())
        self.assertEqual(trial_directory.parent, pathlib.Path(base_dir) / unique_nickname)
        self.assertTrue(training)
        filepath_tools.rm_tree(trial_directory)

    def test_get_trial_directory_new_train_error(self):
        base_dir = pathlib.Path('.testing')
        unique_nickname = 'new_trial/subdir_non_existing'
        self.assertRaises(ValueError,
                          filepath_tools.create_or_load_trial,
                          trial_name=unique_nickname,
                          params={},
                          trials_directory=base_dir,
                          write_summary=False)

    def test_get_trial_directory_load_existing(self):
        base_dir = pathlib.Path('.testing')
        unique_nickname = 'new_trial/subdir'
        supposed_trial_directory = base_dir / unique_nickname
        params_filename = base_dir / unique_nickname / 'params.json'
        expected_params = filepath_tools.get_default_params()
        expected_params['a'] = 2
        with params_filename.open("w") as params_file:
            json.dump(expected_params, params_file)
        supposed_trial_directory.mkdir(parents=True, exist_ok=True)

        trial_directory, loaded_params, training = filepath_tools.create_or_load_trial(trial_name=unique_nickname,
                                                                                       write_summary=False,
                                                                                       trials_directory=base_dir)
        self.assertEqual(supposed_trial_directory, trial_directory)
        self.assertFalse(training)
        self.assertEqual(loaded_params, expected_params)


if __name__ == '__main__':
    unittest.main()
