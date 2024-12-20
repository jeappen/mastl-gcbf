import functools as ft
from typing import Optional, Type, Callable

import flax.linen as nn
import jax
from jax import numpy as jnp

from gcbfplus.env.base import MultiAgentEnv
from gcbfplus.nn.planner_nn import EulerOdeNet
from gcbfplus.nn.utils import AnyFloat
from gcbfplus.stl.utils import PLANNER_CONFIG, Goal
from gcbfplus.utils.graph import GraphsTuple
from .base import MultiAgentPlanner, Planner
from ....nn.utils import default_nn_init


class StateInputODEPlanner(nn.Module):
    """Neural planner for planning next goal given the current goal/state"""
    # base_cls: Type[GNN]
    head_cls: Type[nn.Module]
    _ng: int
    max_difference: float = 10.0

    @nn.compact
    def __call__(self, obs: AnyFloat, *args, **kwargs) -> Goal:
        """Predicts the change in the goal given the current goal or agent state"""
        # x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        # x = obs
        assert isinstance(obs, AnyFloat), "obs needs to be an AnyFloat!"
        x = self.head_cls()(obs)
        # NOTE: tanh may cause the gradient to vanish over time
        x = nn.tanh(nn.Dense(self._ng, kernel_init=default_nn_init(), name="OutputDenseGoals")(x)) * self.max_difference
        # x = nn.Dense(self._ng, kernel_init=default_nn_init(), name="OutputDenseGoals")(x)
        return x + obs[:, :self._ng]  # Assuming the first _ng columns are the goal in case of embedding input


class ODEPlanner(Planner):
    """Neural planner for planning paths"""

    def __init__(self, params=None, name=''):
        super().__init__(params)
        self.planner_head = ft.partial(
            EulerOdeNet,
            goal_space_dim=self.params.get('goal_dim', 2),
            hid_sizes=self.params.get('hidden_arch', (32, 32)),
            arch_kwargs={'hidden_arch': self.params.get('hidden_arch', [32, 32])},
            max_difference=None,  # Limit change in StateInputODEPlanner
            # act=nn.relu,
            # act_final=False,
            name='PlannerHead' + name
        )
        self.net = StateInputODEPlanner(head_cls=self.planner_head, _ng=self.params.get('goal_dim', 2),
                                        max_difference=self.params.get('max_difference', 10.0),
                                        name='ODEPlannerHead' + name)
        # self.model = EulerOdeNet()

    def plan(self, params, start, plan_length=10, cbf=None, cbf_params=None, **kwargs):
        """Plan a path from start to satisfy STL constraints"""

        def scan_input_fn(start_arg, _):
            output = self.net.apply(params, start_arg)
            return output, output  # Carry and output are the same

        _, plan = jax.lax.scan(scan_input_fn, start, None, length=plan_length)

        return plan[None]  # Equivalent to unsqueeze(0)


class ODEFeaturePlanner(Planner):
    """ODE planner with a (constant) feature vector as part of the input"""

    def __init__(self, params=None, name=''):
        super().__init__(params)

        self.planner_head = ft.partial(
            EulerOdeNet,
            goal_space_dim=self.params.get('goal_dim', 2) + self.params.get('ode_ip_feature_size', 0),
            hid_sizes=self.params.get('hidden_arch', (32, 32)),
            arch_kwargs={'hidden_arch': self.params.get('hidden_arch', [32, 32])},
            max_difference=None,  # Limit change in StateInputODEPlanner
            # act=nn.relu,
            # act_final=False,
            name='PlannerHead' + name
        )
        self.net = StateInputODEPlanner(head_cls=self.planner_head, _ng=self.params.get('goal_dim', 2),
                                        max_difference=self.params.get('max_difference', 10.0),
                                        name='ODEFeaturePlannerHead' + name)
        # self.model = EulerOdeNet()

    def plan(self, params, start_w_embedding, plan_length=10, cbf=None, cbf_params=None, **kwargs):
        """Plan a path from start to satisfy STL constraints"""
        plan = []
        start = start_w_embedding[:, :self.params['goal_dim']]
        embedding = start_w_embedding[:, self.params['goal_dim']:]
        for i in range(plan_length):
            # Keep the embedding constant
            start = self.net.apply(params, jax.lax.concatenate([start, embedding], dimension=1))
            plan.append(start)
        return jnp.stack(plan)[None]  # Equivalent to unsqueeze(0)


class MAODEPlanner(MultiAgentPlanner):
    """Run the same ODE planner for each agent. Keeping the function signature same as the other planners for compatibility."""

    def __init__(self, num_agents: int, node_dim: int = None, edge_dim: int = None, phi_dim: int = None,
                 action_dim: int = 2, goal_dim: int = 2,
                 planner_config: Optional[dict] = None, filter_state: Optional[Callable] = None, **kwargs):
        """Initialize the planner

        :param num_agents: Number of agents
        :param node_dim: Dimension of the node features
        :param edge_dim: Dimension of the edge features
        :param action_dim: Dimension of the action space
        :param goal_dim: Dimension of the goal space
        :param planner_config: Configuration for the planner
        :param filter_state: Function to filter the state by extracting the goal
                            before passing to the planner or stl functions"""

        super(MAODEPlanner, self).__init__(
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            goal_dim=goal_dim
        )
        # Use the default config
        self.planner_config = PLANNER_CONFIG.copy()
        if planner_config is not None:
            self.planner_config.update(planner_config)
        if filter_state is not None:
            self.filter_state = filter_state
        else:
            self.filter_state = lambda x: x

        self.ode_planner = ODEPlanner(dict(goal_dim=goal_dim, hidden_arch=self.planner_config['hidden_arch']))
        self.net = self.ode_planner.net  # To get the model parameters

    def init(self, key, obs: GraphsTuple, n_agents: int, **kwargs):
        """Initialize the planner with the current state"""
        # Get the initial state
        start = self.filter_state(obs.type_states(MultiAgentEnv.AGENT, n_type=self.num_agents))
        return self.net.init(key, start, n_agents)

    def forward(self, params, obs: GraphsTuple, plan_length=None, **kwargs) -> AnyFloat:
        """
        Get the goal trajectory for the input states.

        Parameters
        ----------
        params: Params
            parameters for the planner
        obs: GraphsTuple,
            batched data using Batch.from_data_list().

        Returns
        -------
        actions: (bs x N x plan_length x goal_dim)
            goals for all agents over entire plan
        """
        if plan_length is None:
            plan_length = self.planner_config['plan_length']
        # Get the starting states for all agents
        start = self.filter_state(obs.type_states(MultiAgentEnv.AGENT, n_type=self.num_agents))
        # Return the full plan (bs x N x plan_length x goal_dim)
        full_plan = self.ode_planner.plan(params, start, plan_length=plan_length)

        return full_plan

    def forward_ode(self, params, start_goals, plan_length=1, **kwargs) -> AnyFloat:
        """
        Get the ODE goal trajectory for the input states (not graph observation).

        Parameters
        ----------
        params: Params
            parameters for the planner
        start_goals: Array,
            batched data using Batch.from_data_list().

        Returns
        -------
        actions: (bs x N x plan_length x goal_dim)
            goals for all agents over entire plan
        """
        # Return the full plan (bs x N x plan_length x goal_dim)
        return self.ode_planner.plan(params, start_goals, plan_length=plan_length)
