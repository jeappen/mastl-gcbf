import functools as ft
import jax
import unittest

from gcbfplus.env.wrapper import AsyncNeuralSTLWrapper


class TestEnvLoading(unittest.TestCase):
    """Tests for the loading of the environment."""
    env_name = 'DubinsCar'
    num_agents = 1
    area_size = 4
    # Rest Arbitrary
    spec_len = 18
    max_step = 1000

    def setUp(self):
        pass

    def test_loading_env(self):
        """Tests valid loading of the environment."""
        stl_wrapper = AsyncNeuralSTLWrapper  # wrapper_fn
        # Sub test for the specs seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9
        for i in range(2, 5):
            for spec in ['seq', 'mseq']:
                with self.subTest(spec=f'{spec}{i}'):
                    env = self._load_env(stl_wrapper, f'{spec}{i}')
                    self.assertTrue(env is not None)


    def _load_env(self, stl_wrapper=None, spec=None):
        """Helper function to load the environment."""
        if stl_wrapper is not None and spec is not None:
            part_stl_wrapper = ft.partial(stl_wrapper, spec=spec, spec_len=self.spec_len, max_step=self.max_step,
                                          goal_set_args={'dont_shuffle': True})
        else:
            part_stl_wrapper = None
        from gcbfplus.env import make_env
        env = make_env(
            env_id=self.env_name,
            num_agents=self.num_agents,
            area_size=self.area_size,
            max_step=self.max_step,
            wrapper_fn=part_stl_wrapper
        )

        return env


if __name__ == '__main__':
    unittest.main()