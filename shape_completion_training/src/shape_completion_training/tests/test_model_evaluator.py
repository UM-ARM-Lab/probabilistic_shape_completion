from unittest import TestCase
import numpy as np

from shape_completion_training.plausible_diversity.observation_model import observation_likelihood


class TestObservationModel(TestCase):
    def test_observation_probability(self):
        vg_expected = np.zeros((3,3,3))
        vg_expected[1,1,1] = 1.0
        vg_expected[2,1,1] = 1.0
        vg_expected[1,1,0] = 1.0
        vg_observed = vg_expected.copy()
        vg_observed[2,2,2] = 1.0
        p_self = observation_likelihood(vg_expected, vg_expected)
        p_other = observation_likelihood(vg_observed, vg_expected)

        self.assertGreater(p_self, p_other)


