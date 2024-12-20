import logging
from tqdm.auto import tqdm as tqdm

import matplotlib.pyplot as plt
import numpy as np

from ds.stl import STL, RectAvoidPredicate, RectReachPredicate, StlpySolver
from ds.utils import default_tensor
from typing import Optional, Dict, Any

from ds.stl import default_tensor
from datetime import datetime
from ray import train as ray_train
from ray.tune.stopper import Stopper, CombinedStopper, TrialPlateauStopper
import os
from typing import Dict, Any

import jax
import numpy as np
import yaml
from jaxtyping import Array, Float
from ray.tune.stopper import Stopper, CombinedStopper, TrialPlateauStopper

import gcbfplus.utils.configs as gcbf_module_configs

from flax import core, struct
from jaxtyping import Array, Bool, Float, Int, Shaped
from typing import Dict, TypeVar, Any, List
from numpy import ndarray

# environment types
Goal = Float[Array, 'goal_dim']
Goals = Float[Array, 'num_agents goal_dim']

STL_INFO_KEYS = ['max_path_score', 'min_path_score', 'mean_path_score', 'finish_rate', 'TtR']
STL_PY_NAME = "stlpy"
MAMPS_NAME = "mamps"

LARGE_HARDNESS = 100
SMALL_HARDNESS = 3

INIT_STL_SCORE = -3.0  # Initial STL score as auxiliary node feature
DEFAULT_LOGGED_GRAD_NORM = -1.0  # Default logged gradient norm
DEFAULT_TtR = -1.0
STL_EVAL_TOLERANCE = 2*1e-2  # Tolerance for satisfying STL constraints


def load_yaml(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def load_config(currentpath: str = None) -> dict:
    config = yaml.safe_load(
        open(os.path.join(
            *((list(gcbf_module_configs.__path__) if currentpath is None else currentpath) + ['default_config.yaml'])),
            'r'))
    training_config = config['training_params']
    planner_config = config['planner_params']
    env_config = config['env_params']
    dir_config = config['dir_params']

    return {'training': training_config, 'planner': planner_config, 'env': env_config, 'dir': dir_config}


CONFIGS = load_config()
TRAINING_CONFIG = CONFIGS['training']
PLANNER_CONFIG = CONFIGS['planner']
ENV_CONFIG = CONFIGS['env']
DIR_CONFIG = CONFIGS['dir']


def get_feature_dim(features: list, goal_dim: int, action_dim: int, time_dim: int) -> int:
    """
    Get the feature dimension for the planner and a dictionary of feature to dimension mapping.
    """
    dim = 0
    feature_to_dim = {'u_ref': action_dim, 'current_time': time_dim, 'goals': goal_dim, 'states': goal_dim}
    if features is not None:
        for feature in features:
            if feature in feature_to_dim:
                dim += feature_to_dim[feature]
    return dim, feature_to_dim


import jax.numpy as jnp


# @jax.jit
def get_loss_ind_from_step(step: tuple, params: dict = None, update_step_ind=2) -> int:
    """Get the index of the loss from the step (JAX-compatible).

    switch between the losses based on the update step
    Eg. if update_proportions = [1, 2, 3], then the loss index will be 0 for the first update_duration steps,
    1 for the next 2*update_duration steps, and 2 for the next 3*update_duration steps, then back to 0, and so on.
    Cycle through loss indices based on update_step

    :arg step: tuple, (outer_step, inner_epoch, inner_step)
    :arg params: dict, planner configuration
    :arg update_step_ind: int, index of the update step in the step tuple (0 for outer_step, 1 for inner_epoch, 2 for inner_step)
    """

    if params is None:
        params = PLANNER_CONFIG

    update_duration = params['slow_update_duration']
    update_proportions = jnp.array(params['slow_update_proportions'])
    # default_loss_ind =jnp.nonzero(update_proportions)[0][0]
    mask = (update_proportions != 0)
    default_loss_ind = jnp.argmax(mask)

    # Ensure update_proportions are non-negative integers
    # assert jnp.all(update_proportions >= 0) & jnp.all(update_proportions == update_proportions)

    update_step = step[update_step_ind]
    number_of_intervals = jnp.sum(update_proportions)

    total_steps = update_duration * number_of_intervals
    remaining_step = update_step % total_steps

    cumulative_proportion = jnp.cumsum(update_proportions) * update_duration
    loss_ind = jnp.argmax(remaining_step < cumulative_proportion)

    # Skip achievable loss (index 2) if step[0] less than achievable_warmup_period
    loss_ind = jax.lax.cond((step[0] < params['achievable_warmup_period']) & (loss_ind == 2).astype(bool),
                            lambda _: default_loss_ind,
                            lambda loss_ind_arg: loss_ind_arg, loss_ind)

    return loss_ind.astype(int)  # Convert JAX DeviceArray to Python int


# Stopper for the training
class NanLossStopper(Stopper):
    """Stop a trial if any loss is NaN."""

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        # Check if any loss key is nan
        for key in result:
            if key.startswith("loss/") and np.isnan(result[key]):
                return True
        return False

    def stop_all(self) -> bool:
        return False


class TimeoutStopper(Stopper):
    """Stop a trial if it runs for too long."""

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        # Check if timeout
        if "time_total_s" not in result:
            return False
        return result["time_total_s"] > TRAINING_CONFIG['timeout_value']

    def stop_all(self) -> bool:
        return False


FinalStopper = CombinedStopper(  # To stop the training early
    NanLossStopper(),
    TrialPlateauStopper(
        metric="loss/real_stl_loss",
        mode="min",
        metric_threshold=0.00),
    TimeoutStopper()
)
