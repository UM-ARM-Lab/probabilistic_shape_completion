from unittest import TestCase
from setup_data_for_unit_tests import load_test_files
from shape_completion_training.voxelgrid import bounding_box


class Test(TestCase):
    def test_get_aabb_returns_correct_shape_and_no_repeats(self):
        vg_orig = load_test_files()[0]
        bounds = bounding_box.get_aabb(vg_orig)
        self.assertEqual((8, 3), bounds.shape)

        pts = []
        for pt in bounds:
            self.assertFalse(tuple(pt) in pts)
            pts.append(tuple(pt))

        for vals in bounds.transpose():
            self.assertEqual(2, len(set(vals)))
