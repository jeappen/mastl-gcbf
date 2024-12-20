import functools as ft
import unittest

from gcbfplus.stl.utils import get_loss_ind_from_step  # Assuming the function is in your_module.py


class TestGetLossIndFromStep(unittest.TestCase):
    """Tests for the get_loss_ind_from_step function cycling through different losses based on the update step."""

    def setUp(self):
        self.PLANNER_CONFIG = {
            'slow_update_duration': 10,
            'slow_update_proportions': [1, 2, 3],
            'achievable_warmup_period': 0,
        }
        self.PLANNER_CONFIG2 = {
            'slow_update_duration': 10,
            'slow_update_proportions': [1, 1, 1],
            'achievable_warmup_period': 100
        }
        # Configuration with a no update proportion for the first loss
        self.PLANNER_CONFIG3 = {
            'slow_update_duration': 10,
            'slow_update_proportions': [0, 1, 1],
            'achievable_warmup_period': 100
        }

    def test_within_intervals(self):
        """Tests valid update_step values within the defined intervals."""
        self.assertEqual(0, get_loss_ind_from_step((0, 0, 5), self.PLANNER_CONFIG))  # Flipped arguments
        self.assertEqual(1, get_loss_ind_from_step((0, 0, 15), self.PLANNER_CONFIG))
        self.assertEqual(2, get_loss_ind_from_step((0, 0, 35), self.PLANNER_CONFIG))

        self.assertEqual(0, get_loss_ind_from_step((0, 0, 5), self.PLANNER_CONFIG2))  # Flipped arguments
        self.assertEqual(1, get_loss_ind_from_step((0, 0, 15), self.PLANNER_CONFIG2))
        self.assertEqual(2, get_loss_ind_from_step((100, 0, 25), self.PLANNER_CONFIG2))

        self.assertEqual(1, get_loss_ind_from_step((0, 0, 5), self.PLANNER_CONFIG3))  # Flipped arguments
        self.assertEqual(2, get_loss_ind_from_step((100, 0, 15), self.PLANNER_CONFIG3))
        self.assertEqual(1, get_loss_ind_from_step((100, 0, 25), self.PLANNER_CONFIG3))

    def test_interval_boundaries(self):
        """Tests update_step values exactly at interval boundaries."""
        self.assertEqual(1, get_loss_ind_from_step((0, 0, 10), self.PLANNER_CONFIG))  # End of 1st interval
        self.assertEqual(2, get_loss_ind_from_step((0, 0, 30), self.PLANNER_CONFIG))  # End of 2nd interval
        self.assertEqual(0, get_loss_ind_from_step((0, 0, 60), self.PLANNER_CONFIG))  # End of 3rd interval

        self.assertEqual(1, get_loss_ind_from_step((0, 0, 10), self.PLANNER_CONFIG2))  # End of 1st interval
        self.assertEqual(0, get_loss_ind_from_step((0, 0, 30), self.PLANNER_CONFIG2))  # End of 2nd interval
        self.assertEqual(0, get_loss_ind_from_step((0, 0, 60), self.PLANNER_CONFIG2))  # End of 3rd interval

        self.assertEqual(2, get_loss_ind_from_step((100, 0, 10), self.PLANNER_CONFIG3))  # End of 1st interval
        self.assertEqual(2, get_loss_ind_from_step((100, 0, 30), self.PLANNER_CONFIG3))  # End of 2nd interval
        self.assertEqual(1, get_loss_ind_from_step((0, 0, 60), self.PLANNER_CONFIG3))  # End of 3rd interval

    def test_exceeding_total_duration(self):
        """Tests update_step values that exceed the total update duration."""
        self.assertEqual(0, get_loss_ind_from_step((0, 0, 65), self.PLANNER_CONFIG))  # End of 1st interval

    def test_warmup(self):
        """Tests valid update_step values within the defined intervals."""
        self.assertEqual(2, get_loss_ind_from_step((10, 0, 35), self.PLANNER_CONFIG))  # Flipped arguments
        self.assertEqual(2, get_loss_ind_from_step((200, 0, 35), self.PLANNER_CONFIG))
        self.assertEqual(2, get_loss_ind_from_step((300, 0, 35), self.PLANNER_CONFIG))

        self.assertEqual(0, get_loss_ind_from_step((10, 0, 25), self.PLANNER_CONFIG2))  # Flipped arguments
        self.assertEqual(2, get_loss_ind_from_step((200, 0, 25), self.PLANNER_CONFIG2))
        self.assertEqual(2, get_loss_ind_from_step((300, 0, 25), self.PLANNER_CONFIG2))

        self.assertEqual(1, get_loss_ind_from_step((0, 0, 15), self.PLANNER_CONFIG3))
        self.assertEqual(2, get_loss_ind_from_step((100, 0, 15), self.PLANNER_CONFIG3))
        self.assertEqual(2, get_loss_ind_from_step((100, 0, 30), self.PLANNER_CONFIG3))

    def test_default_config(self):
        """Tests the function's behavior when no planner_config is provided."""
        # Assuming PLANNER_CONFIG is the default configuration
        # self.assertEqual(get_loss_ind_from_step((0, 0, 0)), 0)
        self.assertIsNotNone(get_loss_ind_from_step((0, 0, 0)))
        # ... more tests with different update_step values ...

    def test_different_update_index(self):
        """Tests the function's behavior when a different update_step_ind is provided."""
        _get_loss_ind_from_step = ft.partial(get_loss_ind_from_step, update_step_ind=0)
        self.assertEqual(1, _get_loss_ind_from_step((10, 0, 35), self.PLANNER_CONFIG))  # Flipped arguments
        self.assertEqual(1, _get_loss_ind_from_step((20, 0, 35), self.PLANNER_CONFIG))
        self.assertEqual(2, _get_loss_ind_from_step((30, 0, 35), self.PLANNER_CONFIG))

        self.assertEqual(1, _get_loss_ind_from_step((10, 0, 25), self.PLANNER_CONFIG2))  # Flipped arguments
        self.assertEqual(0, _get_loss_ind_from_step((20, 0, 25), self.PLANNER_CONFIG2))
        self.assertEqual(0, _get_loss_ind_from_step((30, 0, 25), self.PLANNER_CONFIG2))
        self.assertEqual(2, _get_loss_ind_from_step((110, 0, 25), self.PLANNER_CONFIG2))

        self.assertEqual(1, _get_loss_ind_from_step((0, 0, 15), self.PLANNER_CONFIG3))
        self.assertEqual(1, _get_loss_ind_from_step((10, 0, 15), self.PLANNER_CONFIG3))
        self.assertEqual(2, _get_loss_ind_from_step((110, 0, 30), self.PLANNER_CONFIG3))
