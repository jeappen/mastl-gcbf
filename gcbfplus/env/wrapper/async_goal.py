"""Holds the AsyncWrapper class. This class is used to wrap a synchronous
planner to make it asynchronous. Each agent changes its goal only when the
previous goal is reached and not based on the global time."""

import jax
import jax.numpy as jnp
import os
from typing import Any
from typing import Tuple

from gcbfplus.utils.graph import GraphsTuple

os.environ["DIFF_STL_BACKEND"] = "jax"
from ds.stl import StlpySolver
import ds.stl_jax
from gcbfplus.env import MultiAgentEnv
from gcbfplus.stl.utils import STL_INFO_KEYS, TRAINING_CONFIG, ENV_CONFIG
from gcbfplus.utils.typing import Action, Cost, Done, Info, Reward

ds.stl.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']
ds.stl_jax.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']  # Set hardness for easier backpropagation

from .wrapper import STLWrapper, NeuralSTLWrapper, FormationWrapper


class AsyncPlannerMixin(object):
    """Mixin for all wrappers with asynchronous planning b/w agents."""

    def process_finished_rollouts_info(self):
        """Returns function to run on rollout to get any finished rollout metrics. Separate function from satisfaction"""

        # Change graphs to states
        filter_goal = jax.vmap(self.filter_state)

        def rollout2score(rollout, changed_goals):
            """Needs to be more complex to handle async planner and unknown goal changes (quite hacky)"""
            goal_space = filter_goal(rollout.type_states_rollout(self.env.AGENT, self.env.num_agents))

            # Need to evaluate a minimum length of self.spec_len, so we sample states when there are too little changes
            # Handle too little goal changes and pull random samples ahead
            # NOTE: ideally prioritize last state
            goals_left = jnp.maximum(self.spec_len - changed_goals.sum(axis=0), 0)
            max_indices = changed_goals.shape[0] - jnp.argmax(changed_goals[::-1, :], axis=0) - 1
            for i in range(self.env.num_agents):
                # Fill any with random samples from end
                changed_goals[max_indices[i]::self.goal_sample_interval, i][::-1][:goals_left[i]] = 1

            # fill any with random samples behind
            goals_left = jnp.maximum(self.spec_len - changed_goals.sum(axis=0), 0)
            for i in range(self.env.num_agents):
                changed_goals[max_indices[i]::-self.goal_sample_interval, i][::-1][:goals_left[i]] = 1

            # Last resort, fill from end and beginning for any more samples
            goals_left = jnp.maximum(self.spec_len - changed_goals.sum(axis=0), 0)
            for i in range(self.env.num_agents):
                changed_goals[max_indices[i]::-1, i][:goals_left[i]] = 1
            goals_left = jnp.maximum(self.spec_len - changed_goals.sum(axis=0), 0)
            for i in range(self.env.num_agents):
                changed_goals[:max_indices[i], i][:goals_left[i]] = 1

            # Filter the change points states with padding (keep the last state of goal_space)
            flattened_indices = jnp.where(changed_goals)
            row_indices = flattened_indices[0]
            col_indices = flattened_indices[1]

            # Extract change points and map to correct size output
            op_subsampled_row_indices = jnp.zeros_like(col_indices, dtype=jnp.int32)

            # Keep track of the last index seen for each column
            last_indices = jnp.full(col_indices.max() + 1, -1, dtype=jnp.int32)

            # Note how to handle different agents
            for i, col in enumerate(col_indices):
                last_indices = last_indices.at[col].set(last_indices[col] + 1)  # Increment index for this column
                op_subsampled_row_indices = op_subsampled_row_indices.at[i].set(last_indices[col])

            change_points = jnp.zeros((self.spec_len,) + goal_space.shape[1:])
            change_points = change_points.at[op_subsampled_row_indices, col_indices].set(
                goal_space[row_indices, col_indices])
            score_dict = self.score_path(change_points, changed_goals)
            if TRAINING_CONFIG['ds_params']['include_start_state']:
                # concat initial state (crucial for loop stlpy spec)
                # But what about trajectories that reach the goal state in the final step?
                change_points = jnp.concatenate([goal_space[:1], change_points], axis=0)
                score_dict2 = self.score_path(change_points, changed_goals)  # Alternate score with start state
                score_dict = self._merge_score_dict([score_dict, score_dict2])
            # Set last state as the last change point (since this may be missing)
            return score_dict

        return rollout2score

    def _merge_score_dict(self, score_dicts):
        """Merge the score dictionaries for each agent based on the highest mean finish rate"""
        max_finish_rate = -1
        max_score_dict = None
        for score_dict in score_dicts:
            finish_rate = score_dict[STL_INFO_KEYS[-2]].mean()
            if finish_rate > max_finish_rate:
                # print(f"Picking score dict with finish rate {finish_rate} and max finish rate {max_finish_rate}")
                max_finish_rate = finish_rate
                max_score_dict = score_dict
        return max_score_dict

    def score_path(self, path, changed_goals=None):
        full_stl_score_dict, path_scores = super()._score_path(path)
        # Add the changed goal information
        # Index of last entry in changed goals
        num_timesteps = changed_goals.shape[0]
        last_change = num_timesteps - jnp.argmax(changed_goals[::-1, :], axis=0)

        last_change_w_nan = jnp.where((jnp.array(path_scores) > 0)[:, 0], last_change, jnp.nan)
        full_stl_score_dict[STL_INFO_KEYS[-1]] = last_change_w_nan

        return full_stl_score_dict

    def _increment_times(self, graph: GraphsTuple):
        """Increment times based on reaching goals"""

        def _check_goal_reached(graph_arg: GraphsTuple):
            """Only check goal reached if global time is a multiple of goal_sample_interval"""
            agent = graph_arg.type_states(type_idx=self.AGENT, n_type=self.num_agents)
            agent_position = self.filter_state(agent)
            goal = graph_arg.type_states(type_idx=self.GOAL, n_type=self.num_agents)
            goal_position = self.filter_state(goal)
            error = goal_position - agent_position
            reached_agents = jnp.linalg.norm(error, axis=-1, keepdims=True) < self.async_reach_radius
            goal_outside = jnp.any((goal_position > self.env._xy_max) | (goal_position < self.env._xy_min), axis=-1,
                                   keepdims=True)
            reached_agents = reached_agents | goal_outside  # HACK: If goal is outside area skip reaching it
            reached_agents = reached_agents.squeeze(1)
            # Change goals only when reached but not too early (wait if needed)
            new_time = graph_arg.current_time + reached_agents * self.time_increment
            new_global_time = graph_arg.global_time + self.time_increment
            new_time = jnp.minimum(new_time, new_global_time // self.goal_sample_interval)  # Do not exceed global time
            new_time = jnp.minimum(new_time, self.spec_len)  # Do not exceed spec length to capture completion
            return graph_arg._replace(current_time=new_time, global_time=new_global_time)

        return jax.lax.cond(graph.global_time % self.goal_sample_interval == 0, _check_goal_reached,
                            lambda graph_arg: graph_arg._replace(
                                global_time=graph_arg.global_time + self.time_increment), graph)

    def step(
            self, graph: GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[Any, Reward, Cost, Done, Info]:
        """Take a step in the environment and log when goals are changed."""

        old_time = graph.current_time
        retval = super().step(graph, action, get_eval_info)
        changed_goal = retval[0].current_time - old_time
        info_dict = retval[-1]
        info_dict['changed_goal'] = changed_goal
        # For Async planner, if current time >= spec_len then last goal reached.
        if ENV_CONFIG.get('vanish_on_end', False):
            # If vanish_on_end is True, then shift agents out of map when done
            agent_done = (graph.current_time >= self.spec_len)
            # Shift each agent based on agent id to a final resting place out of map using range(1, num_agents+1)
            x_displacement = jnp.arange(2, self.num_agents + 2) * self.env.area_size * agent_done
            old_states = retval[0].type_states(MultiAgentEnv.AGENT, self.num_agents)
            total_displacement = jnp.zeros_like(old_states)
            total_displacement = total_displacement.at[:, 0].set(x_displacement)
            new_states = retval[0].type_states(MultiAgentEnv.AGENT, self.num_agents) + total_displacement
            retval = (self._set_new_state(retval[0], new_states),) + retval[1:]

        done = (graph.current_time >= self.spec_len).all() | retval[3]
        return retval[:3] + (done, info_dict)

    def _set_new_state(self, graph_arg: GraphsTuple, new_state):
        """Set the goal based on the current plan and time and return the new graph"""
        new_goal_envstate = self.env.EnvState(new_state, graph_arg.env_states.goal,
                                              graph_arg.env_states.obstacle)
        # Need to calculate new goal features
        return self.get_graph(new_goal_envstate)._replace(current_time=graph_arg.current_time,
                                                          current_plan=graph_arg.current_plan,
                                                          global_time=graph_arg.global_time,
                                                          history=graph_arg.history, aux_nodes=graph_arg.aux_nodes)

    def _score_set_of_histories(self, histories):
        """Score a set of histories based on the STL spec"""
        # Do not subsample during eval for async planner since histories already subsampled
        return super()._score_set_of_histories(histories, subsample=False)

    def _render_extra(self, ax=None, rollout=None, viz_opts=None):
        """Render extra STL information on the plot"""
        super()._render_extra(ax, rollout, viz_opts)
        if viz_opts is None:
            viz_opts = {}
        # If 'changed_goals' present, print change points for Async planner
        if 'changed_goal' in rollout.T_info and viz_opts.get('plot_changed_goal', False):
            state_trajectory = rollout.Tp1_graph.type_states_rollout(MultiAgentEnv.AGENT, self.num_agents)
            bool_changed_goal = rollout.T_info['changed_goal'] == 1
            change_points = state_trajectory[1:][bool_changed_goal]
            ax.scatter(change_points[:, 0], change_points[:, 1], c='r', s=2, marker='x', label='Goal Change Points')
            # Thin dotted line showing the change points
            for i in range(self.num_agents):
                # concat one true to  get the start point to bool_changed_goal
                bool_changed_goal_i = jnp.concatenate([jnp.array([True]), bool_changed_goal[:, i]])
                per_agent_change_points = state_trajectory[:, i][bool_changed_goal_i]
                # ax.plot(state_trajectory[:, i, 0], state_trajectory[:, i, 1], c='k', linestyle=':', linewidth=1)
                ax.plot(per_agent_change_points[:, 0], per_agent_change_points[:, 1]
                        , c='r', linestyle=':', linewidth=1)

    def _render_extra_update(self, ax=None, rollout=None, viz_opts=None, kk=None, safe_text=None):
        """Render extra dynamic STL information on the plot"""
        # Append current time to safe_text
        old_text = safe_text[0]._text
        safe_text[0].set_text(f"{old_text}"
                              f"\nTime: {rollout.Tp1_graph.current_time[kk]}")


class AsyncSTLWrapper(AsyncPlannerMixin, STLWrapper):
    """Allows Asynchronous plan b/w agents. Changes the goal only when reached and not based on a global clock."""

    def __init__(self, *args, async_reach_radius=0.1, **kwargs):
        """Initialize the environment.

        :param async_threshold: Threshold for asynchronous goal change based on agent progress
        """

        super().__init__(*args, **kwargs)
        self.async_reach_radius = async_reach_radius


class AsyncNeuralSTLWrapper(AsyncPlannerMixin, NeuralSTLWrapper):
    """Allows Asynchronous plan b/w agents. Changes the goal only when reached and not based on a global clock."""

    def __init__(self, *args, async_reach_radius=0.1, **kwargs):
        """Initialize the environment.

        :param async_threshold: Threshold for asynchronous goal change based on agent progress
        """

        super().__init__(*args, **kwargs)
        self.async_reach_radius = async_reach_radius


class AsyncFormationWrapper(AsyncPlannerMixin, FormationWrapper):
    """Allows Asynchronous plan b/w agents. Follows a single plan for multiple agents but with a formation"""

    def __init__(self, *args, async_reach_radius=0.1, **kwargs):
        """Initialize the environment.

        :param async_threshold: Threshold for asynchronous goal change based on agent progress
        """

        super().__init__(*args, **kwargs)
        self.async_reach_radius = async_reach_radius


ASYNC_WRAPPER_LIST = [AsyncSTLWrapper, AsyncNeuralSTLWrapper, AsyncFormationWrapper]
