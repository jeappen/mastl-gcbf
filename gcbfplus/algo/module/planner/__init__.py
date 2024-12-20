from .gnn_planner import GNNwODEPlanner, GNNwODEFeaturePlanner
from .nnplanner import MAODEPlanner

planner_map = {
    'gnn-ode': GNNwODEPlanner,
    'gnn-ode-feature': GNNwODEFeaturePlanner,
    'ode': MAODEPlanner
}
