from abc import ABC, abstractmethod

from gcbfplus.nn.utils import AnyFloat
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.typing import PRNGKey


class Planner(ABC):
    """Base class for planning paths"""

    def __init__(self, params=None):
        self.params = {}
        if params is not None:
            assert isinstance(params, dict), "Params is not a dictionary!"
            self.params.update(params)

    @abstractmethod
    def plan(self, params, start, goal, cbf=None, cbf_params=None, **kwargs):
        """Plan a path from start to goal"""
        raise NotImplementedError("plan not implemented")


class MultiAgentPlanner(ABC):
    """Base class for multi-agent planners."""

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, action_dim: int, goal_dim: int):
        super(MultiAgentPlanner, self).__init__()
        self._node_dim = node_dim
        self._edge_dim = edge_dim
        self._action_dim = action_dim
        self._num_agents = num_agents
        self._goal_dim = goal_dim

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def node_dim(self) -> int:
        return self._node_dim

    @property
    def edge_dim(self) -> int:
        return self._edge_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def goal_dim(self) -> int:
        """Dimension of the goal space where the agents are located."""
        return self._goal_dim

    @abstractmethod
    def init(self, key, obs: GraphsTuple, n_agents: int):
        """Initialize the planner with a nominal state."""
        pass

    @abstractmethod
    def forward(self, params, obs: GraphsTuple, **kwargs) -> AnyFloat:
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
        pass

    def forward_sample(self, params, obs: GraphsTuple, key: PRNGKey = None, **kwargs):
        """For any planner that needs to sample a plan."""
        return self.forward(params, obs, **kwargs)

    def forward_ode(self, params, start_goals, plan_length=1, **kwargs) -> AnyFloat:
        """
        Get the goal trajectory for the input states.

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
        # By default, forward_ode is the same as forward
        return self.forward(params, start_goals, plan_length, **kwargs)

    def forward_graph(self, params, obs: GraphsTuple, plan_length=1, **kwargs) -> AnyFloat:
        """
        Get a single step goal trajectory for the input states depending on the current time.

        :param params:
        :param obs:
        :param plan_length:
        :param kwargs:
        :return:
        """
        # By default, forward_graph is the same as forward
        return self.forward(params, obs, plan_length, **kwargs)
