"""Wrappers/Mixins for any environment."""
import jax
import jax.numpy as jnp
import logging
import numpy as np
import os
import pathlib
import re
import string
from jax.lax import stop_gradient
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from typing import Any, Optional
from typing import Tuple

import ds
from gcbfplus.utils.graph import GraphsTuple

os.environ["DIFF_STL_BACKEND"] = "jax"
from ds.stl import StlpySolver
import ds.stl_jax
from gcbfplus.env import MultiAgentEnv
from gcbfplus.env.base import RolloutResult
from gcbfplus.stl.utils import STL_INFO_KEYS, TRAINING_CONFIG, ENV_CONFIG, STL_PY_NAME, MAMPS_NAME, LARGE_HARDNESS, \
    INIT_STL_SCORE, DEFAULT_TtR, STL_EVAL_TOLERANCE
from gcbfplus.utils.typing import Action, Array, Cost, Done, Info, Reward

import functools as ft
from contextlib import contextmanager

# from neural_mastl.planner.pwl_stl_planner import PwlStlPlanner

ds.stl.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']
ds.stl_jax.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']  # Set hardness for easier backpropagation


@contextmanager
def set_stl_jax_hardness(hardness: float):
    """Set the hardness of the softmax function for the duration of the context.
    Useful for making evaluation strict while allowing gradients to pass through during training.
    Note: using ds.stl_jax.set_hardness does not set global HARDNESS for ds.stl_jax

    :param hardness: hardness of the softmax function
    :type hardness: float
    """
    # global HARDNESS
    old_hardness = ds.stl_jax.HARDNESS
    ds.stl_jax.HARDNESS = hardness
    yield
    ds.stl_jax.HARDNESS = old_hardness


class STLMixin(object):
    """Contains functions for loading STL specs"""

    cover_regex = re.compile("m?cover(\d+)_t(\d+)")
    seq_regex = re.compile("m?seq(\d+)_t(\d+)")
    loop_regex = re.compile("m?(\d+)loop(\d+)_t(\d+)")
    branch_regex = re.compile("m?(\d+)branch(\d+)_t(\d+)")
    signal_regex = re.compile("m?(\d+)signal(\d+)_t(\d+)")
    reach_regex = re.compile("reach_t(\d+)")
    # seq for sequential with same goals for all agents, mseq for sequential with different goals for each agent
    SAMPLE_SPECS = ["reach_t100", "cover2_t100", "cover3_t100", "cover4_t100", "seq2_t10", "seq2_t100", "seq3_t100",
                    "seq4_t100", "mseq2_t100"]
    DEFAULT_GOALS = [[0, 0], [2, 2], [2, 0], [0, 2]]

    spec_name = None

    def _map_spec(self, spec: str):
        """Map the spec to a full name"""
        if self.cover_regex.match(spec):
            return "Cover"
        elif self.seq_regex.match(spec):
            return "Sequence"
        elif self.loop_regex.match(spec):
            return "Loop"
        elif self.branch_regex.match(spec):
            return "Branch"
        elif self.signal_regex.match(spec):
            return "Signal"
        elif self.reach_regex.match(spec):
            return "Reach"
        else:
            raise ValueError(f"Spec {spec} not recognized")

    stl_forms = None  # For multiple agent
    logger = logging.getLogger(__name__)

    @property
    def extra_config(self):
        """Extra configuration for the environment from the wrapper."""
        return ENV_CONFIG

    def _load_diff_spec(self, spec: str, goal_list: list = None, agent_id: int = 0):
        """Load STL spec from the diffspec library

        :param spec: name of spec to load
        :type spec: str
        :param goal_list: list of goals to use for different agents
        :type goal_list: list or None
        :param agent_id: agent id (used to decide goal ordering for current spec)
        :type agent_id: int
        """
        from ds.stl_jax import STL, RectReachPredicate

        # STL spec loading
        # We have two options here:
        # 1. High level spec over time intervals
        # 2. Spec over all time

        # Simple sample spec for now
        goal_size = np.array([1, 1]) * ENV_CONFIG['goal_size']
        get_goal_pred = lambda cent, i: STL(RectReachPredicate(np.array(cent), goal_size, f"goal_{i}",
                                                               shrink_factor=TRAINING_CONFIG['ds_params'][
                                                                   'shrink_factor']))
        if goal_list is None:
            goals_to_use = self.DEFAULT_GOALS
        else:
            # Rotate goals for each agent (arbitrary choice)
            goals_to_use = goal_list[agent_id:] + goal_list[:agent_id]
        goal_predicates = list(map(get_goal_pred, goals_to_use, range(len(goals_to_use))))

        if self.reach_regex.match(spec):
            # Single goal reach spec
            # Extract time interval
            time_int = int(self.reach_regex.match(spec).groups()[0])
            stl_form = goal_predicates[0].eventually(0, time_int)
            self.logger.info(f"Loaded reach spec {spec} with time interval {time_int}")
        elif self.cover_regex.match(spec):
            # cover_spec=
            # Extract number of goals and time interval
            num_goals, time_int = map(int, self.cover_regex.match(spec).groups())
            stl_form = goal_predicates[0].eventually(0, time_int)
            for i, goal in enumerate(goal_predicates[1:num_goals]):
                stl_form = stl_form & goal.eventually(0, time_int)
            self.logger.info(f"Loaded cover spec {spec} with {num_goals} goals, time interval {time_int}")
        elif self.seq_regex.match(spec):
            # seq_spec
            # Extract number of goals and time interval
            num_goals, time_int = map(int, self.seq_regex.match(spec).groups())
            interval_length = time_int // num_goals
            # Start from end to get sensible goal predicate order
            stl_form = goal_predicates[num_goals - 1].eventually(max(0, (num_goals - 1) * interval_length), time_int)
            for i, goal in enumerate(reversed(goal_predicates[:num_goals - 1])):
                time_ind = num_goals - 1 - i
                stl_form = goal.eventually(max(0, (time_ind - 1) * interval_length),  # try minus to fix weird bug
                                           time_ind * interval_length) & stl_form
            self.logger.info(f"Loaded seq spec {spec} with {num_goals} goals, time interval {time_int}")

        elif self.loop_regex.match(spec):
            # loop_spec
            # Extract number of loops, goals and time interval
            num_loops, num_goals, time_int = map(int, self.loop_regex.match(spec).groups())
            per_loop = time_int // num_loops
            stl_form = goal_predicates[0].eventually(0, per_loop)
            for i, goal in enumerate(goal_predicates[1:num_goals]):
                stl_form = stl_form & goal.eventually(0, per_loop)
            stl_form = stl_form.always(0, time_int - per_loop)
            self.logger.info(f"Loaded loop spec {spec} with {num_loops} loops, {num_goals} goals,"
                             f" time interval {time_int}, per loop {per_loop}")

            # Old type of loop  # loop_spec = goal_predicates[0] & goal_predicates[1].eventually(0, 1)  # for i in range(1, num_loops):#len(goal_predicates)):  #     loop_spec |= goal_predicates[i] & goal_predicates[(i + 1) % len(goal_predicates)].eventually(0, 1)  # loop_spec = loop_spec.always(1, time_int - 1)  # stl_form = loop_spec

        elif self.signal_regex.match(spec):
            # signal_spec
            # Extract number of loops, goals and time interval
            num_loops, num_goals, time_int = map(int, self.signal_regex.match(spec).groups())
            signal_portion = 2  # 3 for 15 # 5 for 30 time interval
            actual_time = time_int
            time_int -= signal_portion

            per_loop = time_int // num_loops
            stl_form = goal_predicates[0].eventually(0, per_loop)
            for i, goal in enumerate(goal_predicates[1:num_goals]):
                stl_form = stl_form & goal.eventually(0, per_loop)
            loop_invariant1 = stl_form

            loop_invariant2 = goal_predicates[0].eventually(per_loop, 2 * per_loop)
            for i, goal in enumerate(goal_predicates[1:num_goals]):
                loop_invariant2 = loop_invariant2 & goal.eventually(per_loop, 2 * per_loop)
            # loop_invariant2 = stl_form
            stl_form = stl_form.always(0, time_int - per_loop)  # 3 signal means 2 loop and one reach last goal
            # TODO: Get working until
            # Spec to check if first goal visited twice
            goal0_visited = goal_predicates[0].eventually(per_loop + 1, time_int) & goal_predicates[0].eventually(0,
                                                                                                                  per_loop - 1) & \
                            goal_predicates[1].eventually(per_loop, time_int)
            goal0_visited = loop_invariant1 & loop_invariant2 & goal_predicates[0].eventually(int(1.5 * per_loop),
                                                                                              time_int)
            # goal0_visited_twice = goal0_visited & goal0_visited.eventually(per_loop-1, 2*per_loop)
            stl_form = stl_form.until(goal0_visited, 0, 1)

            # Now visit first goal once at the end of loop
            # stl_form = stl_form & goal_predicates[0].eventually(time_int - 1, actual_time - 1)

            # Now visit last goal once
            goal_last = goal_predicates[-1].eventually(actual_time - signal_portion, actual_time)
            stl_form = stl_form & goal_last

            self.logger.info(f"Loaded signal spec {spec} with {num_loops} loops, {num_goals} goals,"
                             f" time interval {time_int}, per loop {per_loop}")

        elif self.branch_regex.match(spec):
            # branch_spec
            # Extract number of branches, goals per branch and time interval
            num_branches, num_goals, time_int = map(int, self.branch_regex.match(spec).groups())
            stl_form_branches = []
            grouped_goals = [goal_predicates[i:i + num_goals] for i in range(0, len(goal_predicates), num_goals)]
            for goals_in_branch in grouped_goals[:num_branches]:
                stl_form = goals_in_branch[0].eventually(0, time_int)
                for i, goal in enumerate(goals_in_branch[1:num_goals]):
                    stl_form = stl_form & goal.eventually(0, time_int)
                stl_form_branches.append(stl_form)
            stl_form = ft.reduce(lambda x, y: x | y, stl_form_branches)

            self.logger.info(f"Loaded branch spec {spec} with {num_branches} branches, {num_goals} goals,"
                             f" time interval {time_int}")
        else:
            # Default fixed loop spec
            # form is the formula goal_1 eventually in 0 to 5 and goal_2 eventually in 0 to 5
            # and that holds always in 0 to 8
            # In other words, the path will repeatedly visit goal_1 and goal_2 in 0 to 13
            stl_form = (goal_predicates[0].eventually(0, 5) & goal_predicates[1].eventually(0, 5)).always(0, 8)

        self.spec_name = self._map_spec(spec)
        print(f"Loaded {self.spec_name} spec: {spec}"
              f"STL spec: {stl_form}")
        return stl_form


from gcbfplus.env.wrapper.base import PlannerWrapper


class STLWrapper(STLMixin, PlannerWrapper):
    """Mixin for all environments."""

    def __init__(self, *args, stl_solver=STL_PY_NAME, spec=None, spec_len=400, device=None, max_step=None,
                 goal_sample_interval=1, plan_length=None, **kwargs):
        """Initialize the environment. choose between MAMPS and STLpy solver."""
        self.max_step = int(spec_len) if max_step is None else max_step
        self.max_episode_steps = self.max_step  # Backward compatibility
        self.spec_len = spec_len
        self.plan_length = self.spec_len if plan_length is None else plan_length
        self.stl_solver = stl_solver

        super().__init__(*args, device=device, max_step=self.max_step, goal_sample_interval=goal_sample_interval,
                         plan_length=self.plan_length, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.spec = f'seq2' if spec is None else spec
        self._init_plan(spec=self.stl_string, agent_goals=self._set_agent_goals())

        self.env._render_extra = self._render_extra  # Since this is just a wrapper, we need to set the render extra

        all_stl_specs = set([stl_form.__repr__() for stl_form in self.stl_forms])
        assert len(all_stl_specs) == 1, f"STL formulas are different: {all_stl_specs}"
        agent_id = 0

        # Look at the AsyncMixIn for how to vary goal asynchronously (based on reaching)
        self.goal_step = self.max_step // self.stl_forms[agent_id].end_time()
        self.logger.info(f"Using {self.__class__.__name__} wrapper with spec {self.stl_string}")

    @property
    def stl_string(self):
        """Get the STL string corresponding to the spec and time length"""
        return f'{self.spec}_t{self.spec_len}'

    def change_uref(self, uref):
        """Change the reference actions."""
        raise NotImplementedError

    def get_all_predicates(self):
        """Get all predicates in the STL spec"""
        return self.stl_form.get_all_predicates()

    def reset(self, key: Array, plan_settings=None):
        """Reset the environment."""
        if plan_settings is None:
            plan_settings = {'time_limit': 15 * 60 // self.env.num_agents,  # 15 min per run
                             'plan_length': self.plan_length}

        retval = super(PlannerWrapper, self).reset(key)
        agent_states = retval.type_states(MultiAgentEnv.AGENT, self.env.num_agents)
        self._init_plan_with_env(agent_states, **plan_settings)
        # Can change plot limits to display STL predicates
        shifted_cents = [x.cent - (x.size / 2) for x in self.get_all_predicates()]
        pred_sizes = [x.size for x in self.get_all_predicates()]
        # Check rectangles around predicates
        self._xy_min = [0, 0]
        self._xy_max = [self.area_size, self.area_size]
        for pred_size, cent in zip(pred_sizes, shifted_cents):
            lbc = cent
            ruc = cent + pred_size
            self._xy_min = np.minimum(self._xy_min, lbc)
            self._xy_max = np.maximum(self._xy_max, ruc)
        self.env._xy_min = self._xy_min
        self.env._xy_max = self._xy_max
        self.path = []  # Have made functional option by using graph.histories
        return self._init_stl_data(retval)

    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False) -> Tuple[
        Any, Reward, Cost, Done, Info]:
        """Take a step in the environment.

        :param graph: current graph
        :type graph: GraphsTuple"""

        graph = self.change_goals(graph)
        retval = super(PlannerWrapper, self).step(graph, action)
        # self.path.append(stop_gradient(retval[0].states))

        info_dict = retval[-1]
        # Add STL specific data to observation
        new_graph, reward, cost, env_done, _ = retval
        # Can change done based on spec satisfaction if using graph.histories feature
        done = (graph.current_time >= self.max_step).all()  # or self.check_spec_satisfaction(self.path)
        new_retval = self._add_stl_data_to_obs(new_graph, graph), reward, cost, done, info_dict
        return new_retval

    def _add_stl_data_to_obs(self, new_graph: GraphsTuple, graph: GraphsTuple):
        """Add any STL specific data to observation"""
        new_graph = new_graph._replace(current_time=graph.current_time, current_plan=graph.current_plan,
                                       global_time=graph.global_time)
        return new_graph  # No STL data to add

    def _score_set_of_histories(self, histories, subsample=True):
        """Score a set of histories based on the STL spec"""
        if subsample:
            subsampled_hist = histories[self.goal_step - 1::self.goal_step]
        else:
            subsampled_hist = histories
        subsampled_hist = subsampled_hist.transpose((1, 0, 2))

        def eval_stl_form(history, stl_form):
            return stl_form.eval(history)

        # Run each STL formula on the subsampled history for that agent
        output = [eval_stl_form(subsampled_hist[i, None], self.stl_forms[i]) for i in range(len(self.stl_forms))]
        return output

    def _score_path(self, path):
        path_scores = []
        tensor_part = jnp.stack(path)
        all_paths_tensor = stop_gradient(tensor_part)
        with set_stl_jax_hardness(LARGE_HARDNESS):
            # Evaluate the STL spec on the path with true values
            path_scores = jnp.array(stop_gradient(self._score_set_of_histories(all_paths_tensor)))
        max_path_score = path_scores.max()
        min_path_score = path_scores.min()
        mean_path_score = path_scores.mean()
        # Whether agent finished spec
        finish_rate = (path_scores > -STL_EVAL_TOLERANCE).squeeze(1)
        # Respect the STL_INFO_KEYS order
        full_stl_score_dict = dict(zip(STL_INFO_KEYS, [max_path_score, min_path_score, mean_path_score, finish_rate]))
        return full_stl_score_dict, path_scores

    def score_path(self, path):
        """Score the path based on the STL spec and add to info dict"""
        stl_score_dict, path_scores = self._score_path(path)
        stl_score_dict[STL_INFO_KEYS[-1]] = jnp.array(DEFAULT_TtR)  # TtR for non-async
        return stl_score_dict

    @property
    def aux_node_dim(self):
        """Get the dimension of the auxiliary nodes if add_aux_features is set"""
        return 1

    def _init_stl_data(self, data):
        """Add any initial STL specific data to observation"""
        dummy_plan = jnp.zeros(
            (1, self.plan_length, self.env.num_agents, self.env.goal_dim))  # to play nice with jax jit
        if len(self.plan) > 0:
            # Set fixed plan if generated (by MILP planner)
            dummy_plan = jnp.array(self.plan).transpose(1, 0, 2)[None]
        data = data._replace(current_time=jnp.tile(self.init_time, self.env.num_agents), current_plan=dummy_plan,
                             global_time=self.init_time)
        return data  # No STL data to add

    def _init_plan_with_env(self, flat_obs, time_limit=20, plan_length=None):
        """Run planner from a given x0 and make a plan for each agent

        :param flat_obs: flattened observation of all agents
        :type flat_obs: torch.Tensor
        :param time_limit: time limit for planner
        """
        if self.stl_solver == STL_PY_NAME:
            solver = StlpySolver(space_dim=self.env.goal_dim)
            # Run for each agent
            x_0s = stop_gradient(flat_obs[:, :self.env.goal_dim])
            total_time = self.max_step if plan_length is None else plan_length  # Must be trajectory length
            self.plan = []
            self.init_time = 0
            import time

            t0 = time.time()
            for agent_id, x_0 in enumerate(x_0s):
                stlpy_form = self.stl_forms[agent_id].get_stlpy_form()
                path, info = solver.solve_stlpy_formula(stlpy_form, x0=x_0.__array__(), total_time=total_time,
                                                        time_limit=time_limit)
                self.logger.info(f"STLpy for agent {agent_id} x0 {x_0} info: {info}")
                if path is None:
                    self.logger.warning(f"Path not found for spec {self.stl_forms[agent_id]} agent {agent_id} x0 {x_0}")
                    path = np.tile(x_0, (total_time + 1, 1))  # Just stay in place
                self.plan.append(path[1:])  # Remove the initial state

            t1 = time.time()
            total_time = t1 - t0
            self.plan_info = {'plan_time': total_time}
        elif self.stl_solver == MAMPS_NAME:
            raise NotImplementedError("finish this")
        else:
            raise NotImplementedError(f"MILP solver {self.stl_solver} not implemented")

    def _set_agent_goals(self):
        """Set agent goals for the environment if needed"""
        return None

    def _init_plan(self, spec=None, agent_goals=None):
        """Init MILP planner for a given spec string"""
        if self.stl_solver == STL_PY_NAME:
            if agent_goals is not None and spec[0] == 'm':
                # Different goals for each agent (use agent_goals)
                goal_list = agent_goals[:, :2].tolist()
                self.stl_forms = []
                for i in range(self.num_agents):
                    self.stl_forms.append(self._load_diff_spec(spec=spec, goal_list=goal_list, agent_id=i))
            else:
                # Same default goals for all agents
                self.stl_form = self._load_diff_spec(spec=spec)
                self.stl_forms = [self.stl_form] * self.num_agents

            print(f"Loaded STLpy spec: {spec}")
        elif self.stl_solver == MAMPS_NAME:
            spec = 'Reach'
            x0s, plots, info, pwl_plan = self.planner.plan(spec, ignore_agent_collision=True)
            # logging.info(f'PWL Plan: {pwl_plan}')
            self.plan = pwl_plan
            self.current_goal = [0] * self.num_agents
            print(f"Loaded MAMPS plan. for spec: {spec}")
        else:
            raise NotImplementedError(f"MILP solver {self.stl_solver} not implemented")

    def _setup_planner(self):
        """Setup any MILP planner module"""
        if self.stl_solver == STL_PY_NAME:
            self.planner = StlpySolver(space_dim=2)
        # elif self.stl_solver == MAMPS_NAME:
        #     raise NotImplementedError()
        #     self.planner = PwlStlPlanner()
        else:
            raise NotImplementedError(f"MILP solver {self.stl_solver} not implemented")

    def process_finished_rollouts(self):
        """Returns function to run on rollout to get satisfaction rates"""

        # Change graphs to states
        filter_goal = jax.vmap(self.filter_state)

        def rollout2score(rollout):
            """Checks min score > 0"""
            return self.score_path(filter_goal(rollout.type_states_rollout(self.env.AGENT, self.num_agents)))[
                STL_INFO_KEYS[1]] > 0

        return rollout2score

    def process_finished_rollouts_info(self):
        """Returns function to run on rollout to get any finished rollout metrics. Separate function from satisfaction"""

        # Change graphs to states
        filter_goal = jax.vmap(self.filter_state)

        def rollout2score(rollout):
            return self.score_path(filter_goal(rollout.type_states_rollout(self.env.AGENT, self.num_agents)))

        return rollout2score

    def _render_extra(self, ax=None, rollout=None, viz_opts=None):
        """Render extra STL information on the plot"""
        goal_color = 'green'
        # To add any ellipses for goals
        # goal_circles = EllipseCollection([0.6 * d_x] * num_goals, [0.6 * d_x] * num_goals,
        #                                  np.zeros(num_goals),
        #                                  offsets=goal_centers * np.array([[d_x, d_y]]), units='x',
        #                                  color=colors_goals,
        #                                  linewidth=2,
        #                                  alpha=0.5,
        #                                  transOffset=ax.transData, zorder=-1)
        # goal_circles.set_edgecolor('black')
        # ax.add_collection(goal_circles)
        # Sort to get consistent order and only unique predicates
        predicates = sorted(set(self.get_all_predicates()))
        shifted_cents = [x.cent - (x.size / 2) for x in predicates]
        pred_sizes = [x.size for x in predicates]

        text_font_opts = dict(size=16, color="k", family="sans-serif", weight="normal",  # transform=ax.transAxes,
                              )
        # Plot rectangles around predicates
        pred_center = []
        for i, (pred_size, cent) in enumerate(zip(pred_sizes, shifted_cents)):
            rect = plt.Rectangle(cent, pred_size[0], pred_size[1], edgecolor='gray', facecolor=to_rgba(goal_color, 0.4),
                                 linewidth=2)
            ax.add_patch(rect)
            center_x = rect.get_x() + rect.get_width() / 2
            center_y = rect.get_y() + rect.get_height() / 2
            # print(f"Center: {center_x}, {center_y}, rect: {rect.get_x()}, {rect.get_y()}, {rect.get_width()}, {rect.get_height()}")
            radius = 0.1

            # Create a circle with faded fill and border
            # circle = plt.Circle((center_x, center_y), radius, edgecolor='gray', facecolor='lightgray', alpha=0.3,
            #                     linewidth=2)
            # ax.add_patch(circle)

            # pred_txt=f"\$\\phi_{i}\$"
            pred_txt = string.ascii_uppercase[i]  # Use letters for predicates
            # Add a text box in the center of the circle
            ax.text(center_x, center_y, pred_txt, ha='center', va='center', fontsize=48, color='gray', alpha=0.9,
                    zorder=2)  # bbox=dict(facecolor='lightgray', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.3))  # if self.env.goal_dim == 2:  #     pred_center.append(ax.text(center_x, center_y, pred_txt, va="center", ha='center', **text_font_opts))  # else:  #     pred_center.append(ax.text2D(center_x, center_y, pred_txt, va="center", ha='center', **text_font_opts))

        # Change the xlim, ylim from (0, self.area_size) to include the STL predicates
        ax.set_xlim(self._xy_min[0], self._xy_max[0])
        ax.set_ylim(self._xy_min[1], self._xy_max[1])

    def _render_extra_update(self, ax=None, rollout=None, viz_opts=None, kk=None, safe_text=None):
        """Render any dynamic STL information on the plot"""
        pass

    def render_video(self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None,
                     dpi: int = 100, **kwargs):
        """Render video with STL predicates"""
        self.env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi, render_extra=self._render_extra,
                              render_extra_update=self._render_extra_update, stl_form=self.stl_form,
                              stl_spec_name=self.spec_name, **kwargs)


class NeuralSTLWrapper(STLWrapper):
    """Wrapper with STL related information but does not run the MILP planner. Used for neural planners."""

    def __init__(self, *args, add_aux_features=False, **kwargs):
        """Initialize the environment.

        :param add_aux_features: Add auxiliary features to the observation from graph.aux_nodes"""

        super().__init__(*args, **kwargs)
        self.add_aux_features = add_aux_features

    def _init_plan_with_env(self, flat_obs, time_limit=20, **kwargs):
        self.logger.info("NeuralSTLWrapper: Not running MILP planner")
        self.path = []
        return None

    def get_graph(self, state: MultiAgentEnv.EnvState, current_time: Optional[Array] = None,
                  current_plan: Optional[Array] = None, history: Optional[Array] = None,
                  aux_nodes: Optional[Array] = None, global_time: Optional[Array] = None):
        """Get the graph for the current state"""
        graph = self.env.get_graph(state)._replace(current_time=current_time, current_plan=current_plan,
                                                   global_time=global_time)
        if self.add_aux_features:
            graph = graph._replace(history=history, aux_nodes=aux_nodes)
        return graph

    def _init_stl_data(self, data: GraphsTuple):
        """Add any initial STL specific data to observation"""
        # Current time along plan for a given agent
        # dummy plan of size (batch, plan_length, num_agents, goal_dim)
        data = super()._init_stl_data(data)
        if not self.add_aux_features:
            # No STL data to add
            return data
        # dummy aux node features
        dummy_aux = INIT_STL_SCORE * jnp.ones((data.nodes.shape[0], self.aux_node_dim))
        init_state = self.filter_state(data.type_states(MultiAgentEnv.AGENT, n_type=self.env.num_agents))
        init_history = jnp.tile(init_state, (self.max_step, 1, 1))  # Use current time to truncate history

        new_data = data._replace(aux_nodes=dummy_aux, history=init_history)
        return new_data

    def _add_stl_data_to_obs(self, new_graph: GraphsTuple, graph: GraphsTuple):
        """Add any STL specific data to observation"""
        new_graph = super()._add_stl_data_to_obs(new_graph, graph)
        if not self.add_aux_features:
            return new_graph
        # Update history
        new_state = self.filter_state(new_graph.type_states(MultiAgentEnv.AGENT, n_type=self.env.num_agents))
        updated_history = graph.history.at[graph.current_time].set(new_state)  # Add latest state
        agent_scores = self._score_set_of_histories(updated_history)
        # Set the STL score as a node feature for node_types agent (Assume first num_agents nodes are agents)
        aux_nodes = graph.aux_nodes.at[jnp.arange(0, self.env.num_agents), 0].set(agent_scores)
        return new_graph._replace(history=updated_history, aux_nodes=aux_nodes)

    def _increment_times(self, graph: GraphsTuple):
        """Increment the current time for each agent"""
        new_global_time = graph.global_time + self.time_increment
        new_time = new_global_time // self.goal_sample_interval
        new_time = jnp.minimum(new_time, self.spec_len)  # Do not exceed spec length
        return graph._replace(current_time=jnp.tile(new_time, self.env.num_agents), global_time=new_global_time)

    def change_goals(self, graph: GraphsTuple, **kwargs):
        """Change goals for neural planner based on current flat observation, not part of wrapper"""
        return self._increment_times(graph)

    def forward(self, graph: GraphsTuple, action):
        graph = self._increment_times(graph)
        return self.env.forward(graph, action)

    def forward_graph(self, graph: GraphsTuple, action, **kwargs):
        if 'current_time' in graph:
            graph = self._increment_times(graph)
        # Careful not to increment twice since env.forward_graph() calls env.forward()
        graph_next = self.env.forward_graph(graph, action, **kwargs)
        if 'current_time' in graph:
            graph_next = graph_next._replace(current_time=graph.current_time)
        return graph_next

    @property
    def extra_info_keys(self):
        return STL_INFO_KEYS


class FormationWrapper(STLWrapper):
    """Wrapper with for environment following a single plan for multiple agents but with a formation"""

    def __init__(self, *args, formation=None, **kwargs):
        """Initialize the environment.

        :param formation: Formation to follow, None for no formation. Supported formations: 'line', 'circle'
        """
        super().__init__(*args, **kwargs)
        self.formation = formation

    def _init_plan_with_env(self, flat_obs, time_limit=20, plan_length=None):
        """Run planner from a given x0 and make a plan for each agent

        :param flat_obs: flattened observation of all agents
        :type flat_obs: jnp.ndarray
        :param time_limit: time limit for planner
        """
        if self.stl_solver == STL_PY_NAME:
            solver = StlpySolver(space_dim=self.env.goal_dim)
            # Run for each agent
            x_0s = stop_gradient(flat_obs[:, :self.env.goal_dim])
            total_time = self.max_step if plan_length is None else plan_length  # Must be trajectory length
            self.plan = []
            self.init_time = 0
            import time

            t0 = time.time()

            # Single plan for all agents
            stlpy_form = self.stl_forms[0].get_stlpy_form()
            # first predicate as start state
            x0 = sorted(self.stl_form.get_all_predicates())[0].cent
            path, info = solver.solve_stlpy_formula(stlpy_form, x0=x0, total_time=total_time - 1,  # -1 for start state
                                                    time_limit=time_limit)

            for agent_id, x_0 in enumerate(x_0s):
                self.logger.info(f"STLpy for agent {agent_id} x0 {x_0} info: {info}")
                if path is None:
                    self.logger.warning(f"Path not found for spec {self.stl_forms[agent_id]} agent {agent_id} x0 {x_0}")
                    path = np.tile(x_0, (total_time + 1, 1))  # Just stay in place
                self.plan.append(path)

            t1 = time.time()
            total_time = t1 - t0
            self.plan_info = {'plan_time': total_time}
        elif self.stl_solver == MAMPS_NAME:
            raise NotImplementedError("finish this")
        else:
            raise NotImplementedError(f"MILP solver {self.stl_solver} not implemented")

    def change_goals(self, graph: GraphsTuple, a_all=None, flat_obs=None, init_path=None, o=None, threshold=None,
                     # Not used
                     **kwargs):
        """Change goals for MILP planner based on current flat observation.

        If formation is not None, then change goals based on formation"""

        # if self.infer_mode:
        #     # No goal change in simple inference mode
        #     # return super().change_goals(a_all, flat_obs, init_path, o, threshold)
        #     return None, None

        if self.formation is None:
            return super().change_goals(graph, a_all, flat_obs, init_path, o, threshold, **kwargs)
        elif self.formation == 'line':
            # Simple priority order among agents and following a line
            def get_new_goal(graph_arg):
                # time based goal change (doing here since using gcbfplus algo and not plangcbf+)
                batch_ind = 0

                # Keep the current plan the same and only increment time based on that.
                # Select different goals for different agents based priority order

                # Select different goals for different agents based on time
                selected_plan = graph_arg.current_plan[batch_ind]
                reshaped_times = jnp.expand_dims(graph_arg.current_time, axis=tuple(range(1, selected_plan.ndim)))
                reshaped_times = jnp.minimum(reshaped_times, selected_plan.shape[0] - 1)  # Do not exceed plan length
                # Transpose to shape (Num_agents, Num_goals, Goal_dim) for easy selection of goals
                original_goals = jnp.take_along_axis(selected_plan.transpose(1, 0, 2), reshaped_times, axis=1).squeeze(
                    1)

                # All agents i except id 0 and N/2 are following agent i-1
                agent_states = self.filter_state(graph_arg.env_states.agent)
                new_goals = jnp.zeros_like(original_goals)
                # new_goals should init with original goals
                new_goals = new_goals.at[0].set(original_goals[0])
                new_goals = new_goals.at[self.num_agents // 2].set(original_goals[self.num_agents // 2])

                # If

                # Now everyone else follows the previous agent
                for i in range(1, self.num_agents // 2):
                    new_goals = new_goals.at[i].set(agent_states[i - 1])
                    # Note: Assumes even number of agents
                    new_goals = new_goals.at[self.num_agents // 2 + i].set(agent_states[self.num_agents // 2 + i - 1])

                # goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)
                new_goal_envstate = self.EnvState(graph_arg.env_states.agent, new_goals, graph_arg.env_states.obstacle)
                # Need to calculate new goal features
                return self.env.get_graph(new_goal_envstate)._replace(current_time=graph_arg.current_time,
                                                                      current_plan=graph_arg.current_plan,
                                                                      global_time=graph_arg.global_time)

            graph = get_new_goal(graph)
            goal_change = (graph.global_time % self.goal_sample_interval == 0)
            graph = jax.lax.cond(goal_change, get_new_goal, lambda graph_: graph_, graph)

            graph = self._increment_times(graph)  # Increment time and change goals
        elif self.formation == 'circle':
            raise NotImplementedError("Finish this")
        else:
            raise NotImplementedError(f"Formation {self.formation} not implemented")

        return graph
