from abc import ABC, abstractmethod
from typing import Optional, Tuple

from gcbfplus.env.base import MultiAgentEnv
from gcbfplus.trainer.data import Rollout
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.typing import Action, Params, PRNGKey, Array


class MultiAgentController(ABC):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            n_agents: int
    ):
        self._env = env
        self._node_dim = node_dim
        self._edge_dim = edge_dim
        self._action_dim = action_dim
        self._n_agents = n_agents

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
    def n_agents(self) -> int:
        return self._n_agents

    @property
    def config(self) -> dict:
        pass

    @property
    def actor_params(self) -> Params:
        pass

    @property
    def planner_params(self) -> Params:
        pass

    @property
    def algo_params(self) -> dict[str, Params]:
        return {'params': self.actor_params} | (
            {} if self.planner_params is None else {'planner_params': self.planner_params})

    def preprocess_graph(self, graph: GraphsTuple, **kwargs) -> GraphsTuple:
        """Preprocess graph before feeding to the algorithm. Can be used to change the plan"""
        return graph  # Default is to do nothing

    def preprocess_graph_with_key(self, graph: GraphsTuple, key: PRNGKey, **kwargs) -> GraphsTuple:
        """Preprocess graph before feeding to the algorithm with a key for sampling"""
        return self.preprocess_graph(graph)  # Default is to do nothing

    @abstractmethod
    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        pass

    @abstractmethod
    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        pass

    @abstractmethod
    def update(self, rollout: Rollout, step: int) -> dict:
        pass

    @abstractmethod
    def save(self, save_dir: str, step: int):
        pass

    @abstractmethod
    def load(self, load_dir: str, step: int):
        pass
