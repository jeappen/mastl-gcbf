import functools as ft
from typing import Optional, Callable
from typing import Type

import flax.linen as nn
import jax.lax

from gcbfplus.env.base import MultiAgentEnv
from gcbfplus.nn.gnn import GNN
from gcbfplus.nn.mlp import MLP
from gcbfplus.nn.utils import AnyFloat
from gcbfplus.stl.utils import Goal
from gcbfplus.stl.utils import PLANNER_CONFIG, get_feature_dim
from gcbfplus.utils.graph import GraphsTuple
from .base import MultiAgentPlanner
from .nnplanner import ODEPlanner, ODEFeaturePlanner
from ....nn.utils import default_nn_init
from ....utils.typing import Action


class Deterministic(nn.Module):
    base_cls: Type[GNN]
    head_cls: Type[nn.Module]
    _nu: int

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Action:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        x = self.head_cls()(x)
        x = nn.tanh(nn.Dense(self._nu, kernel_init=default_nn_init(), name="OutputDense")(x))
        return x


class GNNODEPlanner(nn.Module):
    """Neural planner for planning next goal given the current graph observation"""
    base_cls: Type[GNN]
    head_cls: Type[nn.Module]
    _ng: int
    max_difference: float = 10.0
    filter_state: Optional[Callable] = lambda x: x

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, add_aux_features=False, *args, **kwargs) -> Goal:
        # Change graph node feature with the STL Score as input to planner here
        if add_aux_features:
            # Add the STL score as a feature to the node features
            obs = obs.combine_aux()
        x = self.base_cls()(obs, node_type=MultiAgentEnv.AGENT, n_type=n_agents)
        x = self.head_cls()(x)
        # NOTE: tanh may cause the gradient to vanish over time
        x = nn.tanh(
            nn.Dense(self._ng, kernel_init=default_nn_init(), name="GNNOutputDenseGoals")(x)) * self.max_difference
        # Extract the goal relevant dimensions
        return x + self.filter_state(obs.type_states(MultiAgentEnv.AGENT, n_type=n_agents))


class GNNODEFeaturePlanner(nn.Module):
    """Neural planner for planning next goal and an embedding given the current graph observation"""
    base_cls: Type[GNN]
    head_cls: Type[nn.Module]
    _ng: int
    max_difference: float = 10.0
    embedding_size: int = 32
    filter_state: Optional[Callable] = lambda x: x

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Goal:
        x = self.base_cls()(obs, node_type=MultiAgentEnv.AGENT, n_type=n_agents)
        x = self.head_cls()(x)
        # NOTE: tanh may cause the gradient to vanish over time
        x = nn.tanh(
            nn.Dense(self._ng + self.embedding_size, kernel_init=default_nn_init(), name="GNNOutputDenseGoals")(
                x)) * self.max_difference
        # x = nn.Dense(self._ng, kernel_init=default_nn_init(), name="OutputDenseGoals")(x)
        return jax.lax.concatenate(
            [x[:, :self._ng] + self.filter_state(obs.type_states(MultiAgentEnv.AGENT, n_type=n_agents)),
             x[:, self._ng:]], dimension=1)  #


class GNNPlanner(MultiAgentPlanner):
    """GNN based planner. Returns the goal residual for each agent."""

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, phi_dim: int, action_dim: int, goal_dim: int,
                 gnn_layers: int = 1, planner_config: Optional[dict] = None, **kwargs):
        super(GNNPlanner, self).__init__(
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            goal_dim=goal_dim
        )
        self.time_feature = 1  # Dimension of the time feature

        # self.feat_transformer = Sequential('x, edge_attr, edge_index', [
        #     (ControllerGNNLayer(node_dim=node_dim, edge_dim=edge_dim, output_dim=1024, phi_dim=phi_dim),
        #      'x, edge_attr, edge_index -> x'),
        # ])
        # # Give the planner the time as input (TODO: Also try a recurrent n/w)
        # self.feat_2_action = MLP(in_channels=1024 + action_dim + self.time_feature + self.goal_dim,
        #                          out_channels=goal_dim,
        #                          hidden_layers=(512, 128, 32))

        self.policy_base = ft.partial(
            GNN,
            msg_dim=128,
            hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128),
            hid_size_update=(256, 256),
            out_dim=128,
            n_layers=gnn_layers
        )
        self.policy_head = ft.partial(
            MLP,
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name='PolicyHead'
        )
        self.net = Deterministic(base_cls=self.policy_base, head_cls=self.policy_head, _nu=action_dim)
        self.std = 0.1

    def forward(self, params, obs: GraphsTuple) -> AnyFloat:
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
        actions: (bs x N x 1 x goal_dim)
            goals for all agents over entire plan
        """
        return self.net.apply(params, obs, self.n_agents)


class GNNwODEPlanner(MultiAgentPlanner):
    """Use GNN to predict the change in the goal given the current goal and get a trajectory using the Euler ODE net."""

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, phi_dim: int, action_dim: int, goal_dim: int,
                 planner_config: Optional[dict] = None, filter_state: Optional[Callable] = None,
                 **kwargs):
        """ Initialize the planner.

        :param num_agents: Number of agents
        :param node_dim: Dimension of the node features
        :param edge_dim: Dimension of the edge features
        :param action_dim: Dimension of the action space
        :param goal_dim: Dimension of the goal space
        :param planner_config: Configuration for the planner
        :param filter_state: Function to filter the state by extracting the goal
                            before passing to the planner or stl functions"""
        super(GNNwODEPlanner, self).__init__(
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
        self.time_feature = 1  # Dimension of the time feature

        self.extra_features_dim = 0
        self.extra_features = []
        if self.planner_config['add_extra_features']:
            self.extra_features_dim, self.feature_to_dim = get_feature_dim(self.planner_config['extra_features'],
                                                                           goal_dim, action_dim,
                                                                           self.time_feature)
            self.extra_features = self.planner_config['extra_features']
        # Study the effect of adding the action to the goal prediction and other variations
        self.ode_planner = ODEPlanner(dict(goal_dim=goal_dim, hidden_arch=self.planner_config['hidden_arch']),
                                      name="ODEPlanner")
        # self.ode_planner_model = self.ode_planner.model  # To get the model parameters

        self.policy_base = ft.partial(
            GNN,
            msg_dim=self.planner_config['gnn_msg_dim'],
            hid_size_msg=self.planner_config['gnn_hid_size_msg'],
            hid_size_aggr=self.planner_config['gnn_hid_size_aggr'],
            hid_size_update=self.planner_config['gnn_hid_size_update'],
            out_dim=self.planner_config['gnn_feature_size'] + self.extra_features_dim,
            n_layers=self.planner_config['planner_gnn_layers']
        )
        self.policy_head = ft.partial(
            MLP,
            hid_sizes=self.planner_config['mlp_hid_size'],
            act=nn.relu,
            act_final=False,
            name='GoalPolicyHead'
        )
        # Map from Graph to first goal
        self.net = GNNODEPlanner(base_cls=self.policy_base, head_cls=self.policy_head, _ng=goal_dim,
                                 name="GNNODEPlanner", filter_state=self.filter_state)
        self.std = 0.1

    def init(self, key, obs: GraphsTuple, n_agents: int, **kwargs):
        """Initialize the planner with the current state"""
        params_1 = self.net.init(key, obs, n_agents, capture_intermediates=True,
                                 **kwargs)  # Get start goals for all agents
        start = params_1['intermediates']['__call__'][0]
        params_2 = self.ode_planner.net.init(key, start, n_agents)
        # Combine the parameters
        params = {'params': {}}
        params['params'].update(params_1['params'])
        params['params'].update(params_2['params'])
        return params

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
        # Process the graph observations
        if plan_length is None:
            plan_length = self.planner_config['plan_length']

        assert plan_length > 0, "Plan length must be greater than 0"

        # Get the starting goals for all agents
        start_goals = self.net.apply(params, obs, self.num_agents, **kwargs)
        # Return the full plan (bs x N x plan_length x goal_dim)
        if self.planner_config.get('use_gnn_goal', False):
            full_plan = self.ode_planner.plan(params, start_goals, plan_length=plan_length - 1)
            # concat start goals and full plan
            plan = jax.lax.concatenate([start_goals[None, None], full_plan], dimension=1)
        else:
            plan = self.ode_planner.plan(params, start_goals, plan_length=plan_length)
        return plan

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

        return self.ode_planner.plan(params, start_goals, plan_length=plan_length)

    def forward_graph(self, params, obs: GraphsTuple, plan_length=1, **kwargs) -> AnyFloat:
        """
        Get a single step goal trajectory for the input states depending on the current time.

        :param params:
        :param obs:
        :param plan_length:
        :param kwargs:
        :return:
        """

        use_gnn = (obs.current_time == 0).all()  # If the time is 0, use the GNN to get the goal
        # start_goals = self.filter_state(obs.type_states(MultiAgentEnv.AGENT, n_type=self.num_agents))

        return jax.lax.cond(  # If the time is 0, use the GNN to get the goal
            use_gnn,
            lambda x: self.forward(params, x, plan_length=plan_length, **kwargs),
            lambda x: self.forward_ode(params,
                                       self.filter_state(x.type_states(MultiAgentEnv.AGENT, n_type=self.num_agents)),
                                       plan_length=plan_length, **kwargs),
            obs)


class GNNwODEFeaturePlanner(MultiAgentPlanner):
    """Use GNN to predict the change in the goal given the current goal and get a trajectory using the Euler ODE net.
    Includes a fixed feature extractor from the GNN output to the goal prediction.
    """

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, phi_dim: int, action_dim: int, goal_dim: int,
                 planner_config: Optional[dict] = None, filter_state: Optional[Callable] = None,
                 **kwargs):

        super(GNNwODEFeaturePlanner, self).__init__(
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
        self.time_feature = 1  # Dimension of the time feature

        self.extra_features_dim = 0
        self.extra_features = []
        if self.planner_config['add_extra_features']:
            self.extra_features_dim, self.feature_to_dim = get_feature_dim(self.planner_config['extra_features'],
                                                                           goal_dim, action_dim,
                                                                           self.time_feature)
            self.extra_features = self.planner_config['extra_features']
        # Study the effect of adding the action to the goal prediction and other variations
        self.ode_planner = ODEFeaturePlanner(dict(goal_dim=goal_dim, hidden_arch=self.planner_config['hidden_arch'],
                                                  ode_ip_feature_size=self.planner_config['ode_ip_embedding_size']),
                                             name="ODEFeaturePlanner")
        # self.ode_planner_model = self.ode_planner.model  # To get the model parameters

        self.policy_base = ft.partial(
            GNN,
            msg_dim=self.planner_config['gnn_msg_dim'],
            hid_size_msg=self.planner_config['gnn_hid_size_msg'],
            hid_size_aggr=self.planner_config['gnn_hid_size_aggr'],
            hid_size_update=self.planner_config['gnn_hid_size_update'],
            out_dim=self.planner_config['gnn_feature_size'] + self.extra_features_dim,
            n_layers=self.planner_config['planner_gnn_layers']
        )
        self.policy_head = ft.partial(
            MLP,
            hid_sizes=self.planner_config['mlp_hid_size'],
            act=nn.relu,
            act_final=False,
            name='GoalPolicyHead'
        )
        # Map from Graph to first goal
        self.net = GNNODEFeaturePlanner(base_cls=self.policy_base, head_cls=self.policy_head, _ng=goal_dim,
                                        name="GNNODEFeaturePlanner",
                                        embedding_size=self.planner_config['ode_ip_embedding_size'],
                                        filter_state=self.filter_state)
        self.std = 0.1

    def init(self, key, obs: GraphsTuple, n_agents: int):
        """Initialize the planner with the current state"""
        params_1 = self.net.init(key, obs, n_agents, capture_intermediates=True)  # Get start goals for all agents
        start = params_1['intermediates']['__call__'][0]
        params_2 = self.ode_planner.net.init(key, start, n_agents)
        # Combine the parameters
        params = {'params': {}}
        params['params'].update(params_1['params'])
        params['params'].update(params_2['params'])
        return params

    def forward(self, params, obs: GraphsTuple) -> AnyFloat:
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
        # Process the graph observations

        # Get the starting goals for all agents
        start_goals_and_embeddings = self.net.apply(params, obs, self.num_agents)
        # Return the full plan (bs x N x plan_length x goal_dim)
        full_plan = self.ode_planner.plan(params, start_goals_and_embeddings,
                                          plan_length=self.planner_config['plan_length'])

        return full_plan
