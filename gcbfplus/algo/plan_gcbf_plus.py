import functools as ft
import os
import pickle
from typing import Optional, Tuple, NamedTuple

import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from flax.training.train_state import TrainState
from jaxproxqp.jaxproxqp import JaxProxQP

from gcbfplus.nn.utils import AnyFloat
from gcbfplus.stl.utils import PLANNER_CONFIG, DEFAULT_LOGGED_GRAD_NORM, get_loss_ind_from_step
from gcbfplus.trainer.data import Rollout
from gcbfplus.trainer.utils import compute_norm_and_clip, jax2np, rollout
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.typing import Action, Params, PRNGKey, Array, State
from gcbfplus.utils.utils import merge01, jax_vmap, mask2index, tree_merge, MultipleLossTrainState
from .gcbf_plus import GCBFPlus
from .module.planner import planner_map
from .module.planner.base import MultiAgentPlanner
from ..env import BaseWrapper, MultiAgentEnv


class Batch(NamedTuple):
    graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
    u_qp: Action


class PlanGCBFPlus(GCBFPlus):
    """GCBFPlus with STL planner.

    This class is a wrapper around GCBFPlus with STL planner. It is used to train the
    planner and the CBF+Actor networks together. The planner is trained using the CBF+Actor networks, and the CBF+Actor
    networks can be refined using the planner. The planner is used to generate the goal trajectory for the CBF+Actor"""

    # Used to prevent conflicts with the wandb.config from the cmd line
    command_line_args = ['planner', 'plan_stl_loss_coeff', 'real_stl_loss_coeff', 'achievable_loss_coeff',
                         'plan_length']
    relevant_params = ["goal_sample_interval", "plan_stl_loss_coeff", "achievable_loss_coeff", "real_stl_loss_coeff",
                       "lr", "plan_length", "phi_dim", "planner", "mb_train_subset_size", "disable_plan_sampling",
                       "real_stl_update_interval", "add_aux_features", "achievable_warmup_period",
                       "stop_planner_rollout_grad", "update_interval_mode", "slow_update_duration",
                       "slow_update_proportions"]
    planner: MultiAgentPlanner

    def __init__(
            self,
            env: BaseWrapper,  # Only works with BaseWrapper since goal based envs are required
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            gnn_layers: int,
            batch_size: int,
            buffer_size: int,
            horizon: int = 32,
            lr_actor: float = 3e-5,
            lr_cbf: float = 3e-5,
            alpha: float = 1.0,
            eps: float = 0.02,
            inner_epoch: int = 8,
            loss_action_coef: float = 0.001,
            loss_unsafe_coef: float = 1.,
            loss_safe_coef: float = 1.,
            loss_h_dot_coef: float = 0.2,
            max_grad_norm: float = 2.,
            seed: int = 0,
            params: Optional[dict] = None,
            load_dir: Optional[str] = None,
            load_step: Optional[int] = 1000,
            **kwargs
    ):
        super(PlanGCBFPlus, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            gnn_layers=gnn_layers,
            batch_size=batch_size,
            buffer_size=buffer_size,
            horizon=horizon,
            lr_actor=lr_actor,
            lr_cbf=lr_cbf,
            alpha=alpha,
            eps=eps,
            inner_epoch=inner_epoch,
            loss_action_coef=loss_action_coef,
            loss_unsafe_coef=loss_unsafe_coef,
            loss_safe_coef=loss_safe_coef,
            loss_h_dot_coef=loss_h_dot_coef,
            max_grad_norm=max_grad_norm,
            seed=seed)

        # hyperparams
        default_params = {k: v for k, v in PLANNER_CONFIG.items() if k in self.relevant_params}

        self.params = default_params
        if params is not None:
            # Update default params with user params
            self.params.update(params)

        planner_config = dict(num_agents=self.n_agents,
                              node_dim=self.node_dim,
                              edge_dim=self.edge_dim,
                              phi_dim=self.params['phi_dim'],
                              action_dim=self.action_dim,
                              goal_dim=env.goal_dim,
                              planner_config=self.params.copy(),
                              filter_state=env.filter_state)

        print(f"Using planner: {self.params['planner']}")
        if self.params['planner'] not in ['milp', 'stlpy']:
            self.planner = planner_map[self.params['planner']](**planner_config)
        else:
            raise NotImplementedError("MILP planner not implemented yet")

        # set up planner
        planner_key, key = jr.split(self.key)
        planner_params = self.planner.init(planner_key, self.nominal_graph, self.n_agents,
                                           add_aux_features=self.params['add_aux_features'])
        self.loss_fns_name = ('real_stl', 'plan_stl', 'achievable_loss')
        self.num_losses = len(self.loss_fns_name)
        planner_with_aux = ft.partial(self.planner.forward_sample, add_aux_features=self.params['add_aux_features'])
        if self.params.get('single_optimizer', False):
            #  Single optimizer for a combined planner loss
            planner_optim = optax.adamw(learning_rate=self.params['lr'], weight_decay=1e-3)
            self.planner_optim = optax.apply_if_finite(planner_optim, 1_000_000)
            self.planner_train_state = TrainState.create(
                apply_fn=planner_with_aux,
                params=planner_params,
                tx=self.planner_optim
            )
        else:
            # Set up optimizers for multiple losses on the planner
            self.opts = [
                optax.apply_if_finite(optax.adamw(learning_rate=self.params['lr'], weight_decay=1e-3), 1_000_000)
                for _ in range(self.num_losses)]
            self.planner_train_state = MultipleLossTrainState.create(
                apply_fn=planner_with_aux,
                params=planner_params,
                tx=self.opts,
            )

        self.load_dir = load_dir
        if load_dir is not None:
            print(f"Loading models from {load_dir} at step {load_step}")
            self.load(load_dir, load_step)

    #
    @property
    def config(self) -> dict:
        return {
            'batch_size': self.batch_size,
            'lr_actor': self.lr_actor,
            'lr_cbf': self.lr_cbf,
            'alpha': self.alpha,
            'eps': self.eps,
            'inner_epoch': self.inner_epoch,
            'loss_action_coef': self.loss_action_coef,
            'loss_unsafe_coef': self.loss_unsafe_coef,
            'loss_safe_coef': self.loss_safe_coef,
            'loss_h_dot_coef': self.loss_h_dot_coef,
            'gnn_layers': self.gnn_layers,
            'seed': self.seed,
            'max_grad_norm': self.max_grad_norm,
            'horizon': self.horizon
        } | {k: v
             for k, v in self.params.items() if k not in self.command_line_args} | {'env_config': self._env.config}

    @property
    def planner_params(self) -> Params:
        return self.planner_train_state.params

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, unsafe_mask: Array) -> jnp.ndarray:
        # safe if in the horizon, the agent is always safe
        def safe_rollout(single_rollout_mask: Array) -> Array:
            safe_rollout_mask = jnp.ones_like(single_rollout_mask)
            for i in range(single_rollout_mask.shape[0]):
                start = 0 if i < self.horizon else i - self.horizon
                safe_rollout_mask = safe_rollout_mask.at[start: i + 1].set(
                    ((1 - single_rollout_mask[i]) * safe_rollout_mask[start: i + 1]).astype(jnp.bool_))
                # initial state is always safe
                safe_rollout_mask = safe_rollout_mask.at[0].set(True)
            return safe_rollout_mask

        safe = jax_vmap(jax_vmap(safe_rollout, in_axes=1, out_axes=1))(unsafe_mask)
        return safe

    def _sample_planner_fn(self, planner_function, planner_params: Optional[Params] = None, stop_grad=False, **kwargs):
        """Function to sample a new plan from the planner at the initial time step of the rollout"""

        def true_fn(graph_arg: GraphsTuple) -> GraphsTuple:
            """Get a new plan depending on the current time"""
            # plan_ind = (graph_arg.current_time // self.params['goal_sample_interval']).max()
            initial_time = (graph_arg.current_time % self.params['plan_length'] == 0).all() & (
                    graph_arg.global_time == 0)

            def init_plan_fn(planner_params_arg, _graph_arg: GraphsTuple) -> AnyFloat:
                if stop_grad:
                    # Attempt to reduce backprop length when doing multiple planner samples
                    _graph_arg = jax.lax.stop_gradient(_graph_arg)
                return planner_function(params=planner_params_arg, obs=_graph_arg,
                                        add_aux_features=self.params['add_aux_features'])

            current_eval_plan = jax.lax.cond(initial_time, init_plan_fn, lambda _, _graph: _graph.current_plan,
                                             planner_params, graph_arg)
            # Select different goals for different agents based on time
            return self._set_goal_from_plan(graph_arg, current_eval_plan)

        return true_fn

    def _set_goal_from_plan(self, graph_arg, current_eval_plan):
        """Set the goal based on the current plan and time and return the new graph"""
        batch_ind = 0
        selected_plan = current_eval_plan[batch_ind]
        reshaped_times = jnp.expand_dims(graph_arg.current_time, axis=tuple(range(1, selected_plan.ndim)))
        reshaped_times = jnp.minimum(reshaped_times, selected_plan.shape[0] - 1)
        # Transpose to shape (Num_agents, Num_goals, Goal_dim) for easy selection of goals
        new_goals = jnp.take_along_axis(selected_plan.transpose(1, 0, 2), reshaped_times, axis=1).squeeze(1)
        new_goal_envstate = self._env.EnvState(graph_arg.env_states.agent, new_goals,
                                               graph_arg.env_states.obstacle)
        # Need to calculate new goal features
        return self._env.get_graph(new_goal_envstate, current_time=graph_arg.current_time,
                                   current_plan=current_eval_plan, global_time=graph_arg.global_time,
                                   history=graph_arg.history, aux_nodes=graph_arg.aux_nodes)

    def preprocess_graph(self, graph: GraphsTuple, params: Optional[Params] = None,
                         planner_params: Optional[Params] = None, stop_grad=False) -> GraphsTuple:
        """Change the plan before feeding to the algorithm during evaluation"""
        if planner_params is None:
            planner_params = self.planner_params
        goal_change = (graph.global_time % self.params['goal_sample_interval'] == 0)
        return jax.lax.cond(goal_change,
                            self._sample_planner_fn(self.planner.forward, planner_params, stop_grad=stop_grad),
                            lambda graph_: graph_, graph)

    def preprocess_graph_single_step(self, graph: GraphsTuple, params: Optional[Params] = None,
                                     planner_params: Optional[Params] = None, stop_grad=False) -> GraphsTuple:
        """Change the graph to a single step plan"""
        if planner_params is None:
            planner_params = self.planner_params
        obs = self._env.filter_state(graph.type_states(MultiAgentEnv.AGENT, n_type=self.n_agents))
        single_step_plan = self.planner.forward_ode(params=planner_params, start_goals=obs,
                                                    add_aux_features=self.params['add_aux_features'])
        # Set time to 0 to get the first goal from single_step_plan
        graph = graph._replace(current_time=jnp.zeros_like(graph.current_time))
        return self._set_goal_from_plan(graph, single_step_plan)

    def preprocess_graph_with_key(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None,
                                  planner_params: Optional[Params] = None,
                                  **kwargs) -> GraphsTuple:
        """Uses a deterministic plan for now but giving the option of sampling"""
        if planner_params is None:
            planner_params = self.planner_params
        goal_change = (graph.global_time % self.params['goal_sample_interval'] == 0)
        planner_sample = ft.partial(self.planner_train_state.apply_fn, key=key)

        return jax.lax.cond(goal_change, self._sample_planner_fn(planner_sample, planner_params),
                            lambda graph_, _: graph_, graph, key)

    def act(self, graph: GraphsTuple, params: Optional[Params] = None,
            planner_params: Optional[Params] = None) -> Action:
        """For testing/evaluation"""
        if params is None:
            params = self.actor_train_state.params
        if planner_params is None:
            planner_params = self.planner_params
        action = 2 * self.actor.get_action(params, graph) + self._env.u_ref(graph)
        return action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None,
             planner_params: Optional[Params] = None) -> Tuple[Action, Array]:
        """During training, the key is used to sample actions"""
        if params is None:
            params = self.actor_params
        if planner_params is None:
            planner_params = self.planner_params
        action, log_pi = self.actor_train_state.apply_fn(params, graph, key)
        return 2 * action + self._env.u_ref(graph), log_pi

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=1)
    def update_tgt(self, cbf_tgt: TrainState, cbf: TrainState, tau: float) -> TrainState:
        tgt_params = optax.incremental_update(cbf.params, cbf_tgt.params, tau)
        return cbf_tgt.replace(params=tgt_params)

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_b_u_qp(self, b_graph: GraphsTuple, params) -> Action:
        b_u_qp, bT_relaxation = jax_vmap(ft.partial(self.get_qp_action, cbf_params=params))(b_graph)
        return b_u_qp

    def update_nets(self, rollout: Rollout, safe_mask, unsafe_mask, step: int = None):
        update_info = {}

        b_u_qp = []
        if not self.params.get('skip_actor_update', False):
            # Compute b_u_qp.
            n_chunks = 8
            batch_size = len(rollout.graph.states)
            # Adjust n_chunks to be a factor of batch_size
            n_chunks = jax.lax.while_loop(lambda x: batch_size % x != 0, lambda x: x + 1, n_chunks)
            chunk_size = batch_size // n_chunks

            # t0 = time.time()
            for ii in range(n_chunks):
                graph = jtu.tree_map(lambda x: x[ii * chunk_size: (ii + 1) * chunk_size], rollout.graph)
                b_u_qp.append(jax2np(self.get_b_u_qp(graph, self.cbf_tgt.params)))
            b_u_qp = tree_merge(b_u_qp)

        batch_orig = Batch(rollout.graph, safe_mask, unsafe_mask, b_u_qp)

        for i_epoch in range(self.inner_epoch):
            idx = self.rng.choice(rollout.length, size=rollout.length, replace=False)
            # (n_mb, mb_size)
            batch_idx = np.stack(np.array_split(idx, idx.shape[0] // self.batch_size), axis=0)
            batch = jtu.tree_map(lambda x: x[batch_idx], batch_orig)

            cbf_train_state, actor_train_state, planner_train_state, update_info = self.update_inner(
                self.cbf_train_state, self.actor_train_state, self.planner_train_state, batch,
                (step, i_epoch)
            )
            self.cbf_train_state = cbf_train_state
            self.actor_train_state = actor_train_state
            self.planner_train_state = planner_train_state

        # Update target.
        self.cbf_tgt = self.update_tgt(self.cbf_tgt, self.cbf_train_state, 0.5)

        return update_info

    def sample_batch(self, rollout: Rollout, safe_mask, unsafe_mask):
        if self.buffer.length > self.batch_size and self.unsafe_buffer.length > self.batch_size:
            # sample from memory
            memory, safe_mask_memory, unsafe_mask_memory = self.buffer.sample(rollout.length)
            try:
                unsafe_memory, safe_mask_unsafe_memory, unsafe_mask_unsafe_memory = self.unsafe_buffer.sample(
                    rollout.length * rollout.time_horizon)
            except ValueError:
                unsafe_memory = jtu.tree_map(lambda x: merge01(x), memory)
                safe_mask_unsafe_memory = merge01(safe_mask_memory)
                unsafe_mask_unsafe_memory = merge01(unsafe_mask_memory)

            # append new data to memory
            self.buffer.append(rollout, safe_mask, unsafe_mask)
            unsafe_multi_mask = unsafe_mask.max(axis=-1)
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[unsafe_multi_mask], rollout),
                safe_mask[unsafe_multi_mask],
                unsafe_mask[unsafe_multi_mask]
            )

            # get update data
            # (b, T)
            rollout = tree_merge([memory, rollout])
            safe_mask = tree_merge([safe_mask_memory, safe_mask])
            unsafe_mask = tree_merge([unsafe_mask_memory, unsafe_mask])

            # (b, T) -> (b * T, )
            rollout = jtu.tree_map(lambda x: merge01(x), rollout)
            safe_mask = merge01(safe_mask)
            unsafe_mask = merge01(unsafe_mask)
            rollout_batch = tree_merge([unsafe_memory, rollout])
            safe_mask_batch = tree_merge([safe_mask_unsafe_memory, safe_mask])
            unsafe_mask_batch = tree_merge([unsafe_mask_unsafe_memory, unsafe_mask])
        else:
            # Does not shuffle the data in this branch
            self.buffer.append(rollout, safe_mask, unsafe_mask)
            unsafe_multi_mask = unsafe_mask.max(axis=-1)
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[unsafe_multi_mask], rollout),
                safe_mask[unsafe_multi_mask],
                unsafe_mask[unsafe_multi_mask]
            )

            # (b, T) -> (b * T, )
            rollout_batch = jtu.tree_map(lambda x: merge01(x), rollout)
            safe_mask_batch = merge01(safe_mask)
            unsafe_mask_batch = merge01(unsafe_mask)

        return rollout_batch, safe_mask_batch, unsafe_mask_batch

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        # (n_collect, T)
        unsafe_mask = jax_vmap(jax_vmap(self._env.unsafe_mask))(rollout.graph)
        safe_mask = self.safe_mask(unsafe_mask)
        safe_mask, unsafe_mask = jax2np(safe_mask), jax2np(unsafe_mask)

        rollout_np = jax2np(rollout)
        del rollout
        rollout_batch, safe_mask_batch, unsafe_mask_batch = self.sample_batch(rollout_np, safe_mask, unsafe_mask)

        # inner loop
        update_info = self.update_nets(rollout_batch, safe_mask_batch, unsafe_mask_batch, step)

        return update_info

    def get_qp_action(
            self,
            graph: GraphsTuple,
            relax_penalty: float = 1e3,
            cbf_params=None,
            qp_settings: JaxProxQP.Settings = None,
    ) -> [Action, Array]:
        assert graph.is_single  # consider single graph
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state: State) -> Array:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = self._env.add_edge_feats(graph, new_state)
            return self.get_cbf(new_graph, params=cbf_params)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        h = h_aug(agent_state).squeeze(-1)
        h_x = jax.jacobian(h_aug)(agent_state).squeeze(1)

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
        Lf_h = ei.einsum(h_x, dyn_f, "agent_i agent_j nx, agent_j nx -> agent_i")
        Lg_h = ei.einsum(h_x, dyn_g, "agent_i agent_j nx, agent_j nx nu -> agent_i agent_j nu")
        Lg_h = Lg_h.reshape((self.n_agents, -1))

        u_lb, u_ub = self._env.action_lim()
        u_lb = u_lb[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ub = u_ub[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ref = self._env.u_ref(graph).reshape(-1)

        # construct QP: min x^T H x + g^T x, s.t. Cx <= b
        H = jnp.eye(self._env.action_dim * self.n_agents + self.n_agents, dtype=jnp.float32)
        H = H.at[-self.n_agents:, -self.n_agents:].set(H[-self.n_agents:, -self.n_agents:] * 10.0)
        g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.n_agents)])
        C = -jnp.concatenate([Lg_h, jnp.eye(self.n_agents)], axis=1)
        b = Lf_h + self.alpha * 0.1 * h

        r_lb = jnp.array([0.] * self.n_agents, dtype=jnp.float32)
        r_ub = jnp.array([jnp.inf] * self.n_agents, dtype=jnp.float32)
        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)

        qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
        if qp_settings is None:
            qp_settings = JaxProxQP.Settings.default()
        qp_settings.dua_gap_thresh_abs = None
        solver = JaxProxQP(qp, qp_settings)
        sol = solver.solve()

        assert sol.x.shape == (self.action_dim * self.n_agents + self.n_agents,)
        u_opt, r = sol.x[:self.action_dim * self.n_agents], sol.x[-self.n_agents:]
        u_opt = u_opt.reshape(self.n_agents, -1)

        return u_opt, r

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2, 3))
    def update_inner(
            self, cbf_train_state: TrainState, actor_train_state: TrainState, planner_train_state: TrainState,
            batch: Batch,
            step_info: Tuple[int, int] = None
    ) -> tuple[TrainState, dict]:
        def update_fn(carry, minibatch: Batch):
            """Update the CBF and actor networks using the minibatch"""
            cbf, actor = carry
            # (batch_size, n_agents) -> (minibatch_size * n_agents, )
            safe_mask_batch = merge01(minibatch.safe_mask)
            unsafe_mask_batch = merge01(minibatch.unsafe_mask)

            def get_loss(cbf_params: Params, actor_params: Params) -> Tuple[Array, dict]:
                # get CBF values
                cbf_fn = jax_vmap(ft.partial(self.cbf.get_cbf, cbf_params))
                cbf_fn_no_grad = jax_vmap(ft.partial(self.cbf.get_cbf, jax.lax.stop_gradient(cbf_params)))
                # (minibatch_size, n_agents)
                h = cbf_fn(minibatch.graph).squeeze(-1)
                # (minibatch_size * n_agents,)
                h = merge01(h)

                # unsafe region h(x) < 0
                unsafe_data_ratio = jnp.mean(unsafe_mask_batch)
                h_unsafe = jnp.where(unsafe_mask_batch, h, -jnp.ones_like(h) * self.eps * 2)
                max_val_unsafe = jax.nn.relu(h_unsafe + self.eps)
                loss_unsafe = jnp.sum(max_val_unsafe) / (jnp.count_nonzero(unsafe_mask_batch) + 1e-6)
                acc_unsafe_mask = jnp.where(unsafe_mask_batch, h, jnp.ones_like(h))
                acc_unsafe = (jnp.sum(jnp.less(acc_unsafe_mask, 0)) + 1e-6) / (
                        jnp.count_nonzero(unsafe_mask_batch) + 1e-6)

                # safe region h(x) > 0
                h_safe = jnp.where(safe_mask_batch, h, jnp.ones_like(h) * self.eps * 2)
                max_val_safe = jax.nn.relu(-h_safe + self.eps)
                loss_safe = jnp.sum(max_val_safe) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)
                acc_safe_mask = jnp.where(safe_mask_batch, h, -jnp.ones_like(h))
                acc_safe = (jnp.sum(jnp.greater(acc_safe_mask, 0)) + 1e-6) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)

                # get neural network actions
                action_fn = jax.vmap(ft.partial(self.act, params=actor_params))
                action = action_fn(minibatch.graph)

                # get next graph
                forward_fn = jax_vmap(self._env.forward_graph)
                next_graph = forward_fn(minibatch.graph, action)
                h_next = merge01(cbf_fn(next_graph).squeeze(-1))
                h_dot = (h_next - h) / self._env.dt

                # stop gradient and get next graph
                h_no_grad = jax.lax.stop_gradient(h)
                h_next_no_grad = merge01(cbf_fn_no_grad(next_graph).squeeze(-1))
                h_dot_no_grad = (h_next_no_grad - h_no_grad) / self._env.dt

                # h_dot + alpha * h > 0 (backpropagate to action, and backpropagate to h when labeled)
                labeled_mask = jnp.logical_or(unsafe_mask_batch, safe_mask_batch)
                max_val_h_dot = jax.nn.relu(-h_dot - self.alpha * h + self.eps)
                max_val_h_dot_no_grad = jax.nn.relu(-h_dot_no_grad - self.alpha * h + self.eps)
                max_val_h_dot = jnp.where(labeled_mask, max_val_h_dot, max_val_h_dot_no_grad)
                loss_h_dot = jnp.mean(max_val_h_dot)
                acc_h_dot = jnp.mean(jnp.greater(h_dot + self.alpha * h, 0))

                # action loss
                assert action.shape == minibatch.u_qp.shape
                loss_action = jnp.mean(jnp.square(action - minibatch.u_qp).sum(axis=-1))

                # total loss
                total_loss = (
                        self.loss_action_coef * loss_action
                        + self.loss_unsafe_coef * loss_unsafe
                        + self.loss_safe_coef * loss_safe
                        + self.loss_h_dot_coef * loss_h_dot
                )

                return total_loss, {'loss/action': loss_action,
                                    'loss/unsafe': loss_unsafe,
                                    'loss/safe': loss_safe,
                                    'loss/h_dot': loss_h_dot,
                                    'loss/total': total_loss,
                                    'acc/unsafe': acc_unsafe,
                                    'acc/safe': acc_safe,
                                    'acc/h_dot': acc_h_dot,
                                    'acc/unsafe_data_ratio': unsafe_data_ratio}

            (loss, loss_info), (grad_cbf, grad_actor) = jax.value_and_grad(
                get_loss, has_aux=True, argnums=(0, 1))(cbf.params, actor.params)
            grad_cbf, grad_cbf_norm = compute_norm_and_clip(grad_cbf, self.max_grad_norm)
            grad_actor, grad_actor_norm = compute_norm_and_clip(grad_actor, self.max_grad_norm)
            cbf = cbf.apply_gradients(grads=grad_cbf)
            actor = actor.apply_gradients(grads=grad_actor)
            grad_info = {'grad_norm/cbf': grad_cbf_norm, 'grad_norm/actor': grad_actor_norm}
            return (cbf, actor), grad_info | loss_info

        def update_plan_fn(carry, minibatch: Batch):
            """Update the planner using the minibatch"""
            cbf, actor, planner, step_info, grad_info = carry
            # (batch_size, n_agents) -> (minibatch_size * n_agents, )
            safe_mask_batch = merge01(minibatch.safe_mask)
            unsafe_mask_batch = merge01(minibatch.unsafe_mask)

            # Try running on mb_train_subset_size element per minibatch
            mb_train_subset_size = self.params['mb_train_subset_size']
            minibatch = jtu.tree_map(lambda x: x[:mb_train_subset_size], minibatch)

            def get_all_stl_loss(planner_params: Params, cbf_params: Params, actor_params: Params) -> Tuple[
                Array, dict]:
                agent_id = 0  # Assumes all agents have the same STL spec
                # get planned paths
                planner_fn = jax_vmap(ft.partial(self.planner.forward, planner_params))
                planner_single_step_fn = jax_vmap(ft.partial(self.planner.forward_graph, planner_params))
                # planner_fn_no_grad = jax_vmap(ft.partial(self.planner.forward, jax.lax.stop_gradient(planner_params)))
                # (minibatch_size, n_agents)
                h = planner_fn(minibatch.graph).squeeze(1)  # Consider using in loss
                # (minibatch_size * n_agents,)
                # h = merge01(h)
                # Use the real states from the rollout to get a direct planner loss here

                stl_eval = self._env.stl_forms[agent_id].eval

                # Plan STL Loss (on the planned goals)
                agent_goals = h.transpose(0, 2, 1, 3)

                # Real STL Loss (on the real trajectory)
                # key, self.key = jr.split(self.key)  # Can't change here
                # Sampling from the batch to vary the starting state for the real stl loss
                # skip full rollout and forward graph for a fixed duration
                if self.params['achievable_single_step_target']:
                    # Preprocess the graph for a single step plan
                    preprocess_graph = self.preprocess_graph_single_step
                    new_mb_graph = minibatch.graph._replace(current_time=jnp.zeros_like(minibatch.graph.current_time))
                    plan_shape = minibatch.graph.current_plan.shape
                    # Change N step to single step plan
                    current_plan = jnp.zeros(plan_shape[:2] + (1,) + plan_shape[3:])
                    new_mb_graph = new_mb_graph._replace(current_plan=current_plan)
                    new_minibatch = Batch(new_mb_graph, minibatch.safe_mask, minibatch.unsafe_mask, minibatch.u_qp)
                else:
                    # Use the full planner (may directly use the sampled state's current plan without sampling new)
                    preprocess_graph = self.preprocess_graph
                    new_minibatch = minibatch
                sample_rollout_fn = ft.partial(rollout, env=self._env,
                                               actor=lambda graph, k: (
                                                   self.act(graph, actor_params, planner_params), None),
                                               key=self.key,  # Causes fixed noise
                                               # NOTE: Maybe the times are off for sampling the achievable loss
                                               preprocess_graph=ft.partial(preprocess_graph,
                                                                           planner_params=planner_params,
                                                                           stop_grad=self.params[
                                                                               'stop_planner_rollout_grad']),
                                               max_length=self.params['goal_sample_interval'])
                sampled_rollout = jax_vmap(sample_rollout_fn)(init_graph=new_minibatch.graph)

                # Combine into MBsize x T x N_agents x Goal_dim
                extract_state_from_rollout = jax_vmap(
                    lambda x: x.graph.type_states_rollout(self._env.AGENT, self.n_agents))
                extract_nextstate_from_rollout = jax_vmap(
                    lambda x: x.next_graph.type_states_rollout(self._env.AGENT, self.n_agents))
                extract_state_from_graph = jax_vmap(
                    lambda x: x.graph.type_states(self._env.AGENT, self.n_agents))
                extract_state_from_next_graph = jax_vmap(
                    lambda x: x.next_graph.type_states(self._env.AGENT, self.n_agents))
                sampled_path = extract_nextstate_from_rollout(sampled_rollout)
                # Start from (mb size, T, N_agents, Statedim), vmap twice to get (mb size, T, N_agents, Goal_dim)
                sampled_path = jax_vmap(jax_vmap(self._env.filter_state))(sampled_path)
                # sample final states
                subsampled_path = sampled_path[:, -1::-self.params['goal_sample_interval']]  # Should be a single step
                # Now rollout planner for one step depending on the current time
                # If any times are zero then use GNN for one step else ODE
                # agent_positions = jax_vmap(self._env.filter_state)(extract_state_from_graph(minibatch))
                agent_positions = new_minibatch.graph
                # For now just sample the shuffled batch to get the initial states for achievable loss
                # NOTE: Ideally do a form of balanced sampling to get the initial states and other states in adequate proportion (say 1:2)
                single_step_plan = planner_single_step_fn(agent_positions).squeeze(1)

                # # Subsample the path to get the goal_step
                # subsampled_path = sampled_path[:, goal_step - 1::goal_step]
                # subsampled_path = subsampled_path.transpose((0, 2, 1, 3))  # MBsize x N_agents x T x Goal_dim
                #
                # real_stl_loss = jax.vmap(lambda x: -stl_eval(x).mean())(subsampled_path).mean()
                real_stl_loss = 0  # Not using real stl loss since hard to propagate over long horizons

                # Update the plan output to match STL spec
                # sampled_plan = jax_vmap(lambda x: x.graph.current_plan)(
                #     sampled_rollout)

                plan_stl_loss = jax.vmap(lambda x: -stl_eval(x).mean())(agent_goals).mean()

                if self.params['stop_planner_rollout_grad']:
                    # Note: should not be necessary but doing for good measure
                    subsampled_path = jax.lax.stop_gradient(subsampled_path)

                if self.params['achievable_loss_form'] == 'max':
                    # Max deviation along an agent trajectory
                    achievable_loss = jnp.max(jnp.square(single_step_plan - subsampled_path), axis=-1).mean()
                else:
                    # Average deviation along a trajectory
                    achievable_loss = jnp.mean(jnp.square(single_step_plan - subsampled_path), axis=-1).mean() / 2
                # unsafe region h(x) < 0
                unsafe_data_ratio = jnp.mean(unsafe_mask_batch)

                # total loss (follow self.loss_fns_name order)
                total_loss = jnp.array([self.params['real_stl_loss_coeff'] * real_stl_loss,
                                        self.params['plan_stl_loss_coeff'] * plan_stl_loss,
                                        self.params['achievable_loss_coeff'] * achievable_loss])

                total_loss_sum = jnp.sum(total_loss)

                return (total_loss_sum if self.params['single_optimizer'] else total_loss), {
                    'loss/real_stl': real_stl_loss,
                    'loss/plan_stl': plan_stl_loss,
                    'loss/achievable': achievable_loss,
                    'loss/total': total_loss_sum,
                    'acc/unsafe_data_ratio': unsafe_data_ratio}

            if self.params['single_optimizer']:
                # Update planner with sum of all losses
                (loss, loss_info), (grad_planner,) = jax.value_and_grad(
                    get_all_stl_loss, has_aux=True, argnums=(0,))(planner.params, cbf.params, actor.params, )
                grad_planner, grad_planner_norm = compute_norm_and_clip(grad_planner, self.max_grad_norm)
                planner = planner.apply_gradients(grads=grad_planner)
                grad_info = {'grad_norm/planner': grad_planner_norm}

            else:
                # Update planner with multiple losses
                def get_planner_update(step_info_arg, loss_fns_name, params, max_grad_norm, grad_fn):
                    """Decide which loss to update based on step_info"""

                    def get_grad(loss_ind):
                        def compute_grad(grad_fn_arg):
                            grad_planner = grad_fn_arg(jnp.eye(self.num_losses)[loss_ind])[
                                0]  # Assume i=0 (single update)
                            grad_planner_to_add, grad_planner_norm = compute_norm_and_clip(grad_planner, max_grad_norm)
                            return [grad_planner_to_add], [loss_ind], grad_planner_norm

                        return compute_grad

                    # Simple scheduler alternating between losses based on update intervals.
                    if params['update_interval_mode'] == 'simple':
                        # update_achievable = (step_info_arg[2] % params['achievable_update_interval'] == 0) & (
                        #         step_info_arg[0] > params['achievable_warmup_period'])

                        # achievable_vs_real_fn = lambda _grad_fn: jax.lax.cond(update_achievable, get_grad(2), get_grad(1),
                        #                                                       _grad_fn)
                        # skip real stl update
                        achievable_vs_real_fn = lambda _grad_fn: get_grad(1)(_grad_fn)  # Always update achievable loss

                        # This update order prioritizes real loss over achievable loss (at intersecting update interval)
                        update_real_stl = step_info_arg[0] % params['real_stl_update_interval'] == 0
                        do_update = jax.lax.cond(update_real_stl, get_grad(2), achievable_vs_real_fn, grad_fn)
                    elif params['update_interval_mode'] == 'slow':
                        loss_ind = get_loss_ind_from_step(step_info_arg, params,
                                                          update_step_ind=params['update_interval_step_index'])
                        do_update = get_grad(loss_ind)(grad_fn)

                    return do_update

                f_partial = ft.partial(get_all_stl_loss, cbf_params=cbf.params, actor_params=actor.params)
                primal, grad_fn, loss_info = jax.vjp(f_partial, planner.params, has_aux=True)  # Compute jax.vjp outside

                # Assuming you only want to update with the first loss (i=0)
                (grad_list, loss_ind, grad_planner_norm) = get_planner_update(step_info, self.loss_fns_name,
                                                                              self.params,
                                                                              self.max_grad_norm,
                                                                              grad_fn)

                def log_grad_info(i):
                    """Log the specific grad_norm for the ith loss function"""

                    def get_info(grad_planner_norm):
                        _grad_info = {f'grad_norm/planner_{self.loss_fns_name[_i]}': DEFAULT_LOGGED_GRAD_NORM for _i in
                                      range(self.num_losses)}
                        _grad_info[f'grad_norm/planner_{self.loss_fns_name[i]}'] = grad_planner_norm

                        return _grad_info

                    return get_info

                # Pick the loss and log the grad_norm
                grad_info_new = jax.lax.switch(loss_ind[0], [log_grad_info(_i) for _i in range(self.num_losses)],
                                               grad_planner_norm)
                # Take max val of grad_info_new and grad_info
                grad_info = {k: jnp.maximum(v, grad_info[k]) for k, v in grad_info_new.items()}
                # grad_info = grad_info_new  # Can take last value instead

                grad_dict = {l: g for l, g in zip(loss_ind, grad_list)}
                planner = planner.apply_gradients(grads=grad_dict)
            # Now log step_information
            new_step_info = (*step_info[:2], step_info[2] + 1)
            loss_info['step_info/batch_step'] = new_step_info[2]
            loss_info['step_info/inner_epoch'] = new_step_info[1]
            loss_info['step_info/step'] = new_step_info[0]

            return (cbf, actor, planner, new_step_info, grad_info), grad_info | loss_info

        # Uncomment to train cbf and actor
        # train_state = (cbf_train_state, actor_train_state)
        # (cbf_train_state, actor_train_state), info = lax.scan(update_fn, train_state, batch)
        # Train the planner alone
        if self.params['single_optimizer']:
            grad_info_init = {'grad_norm/planner': DEFAULT_LOGGED_GRAD_NORM}
        else:
            grad_info_init = {f'grad_norm/planner_{self.loss_fns_name[_i]}': DEFAULT_LOGGED_GRAD_NORM for _i in
                              range(self.num_losses)}
        train_state = (cbf_train_state, actor_train_state, planner_train_state, (*step_info, 0), grad_info_init)
        (cbf_train_state, actor_train_state, planner_train_state, _, _), info = lax.scan(update_plan_fn,
                                                                                         train_state,
                                                                                         batch)

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info)
        return cbf_train_state, actor_train_state, planner_train_state, info

    def save(self, save_dir: str, step: int, save_cbf=False):
        """Save planner, CBF, and Actor to save_dir/step/. Only planner is saved by default."""
        model_dir = os.path.join(save_dir, str(step))
        if save_cbf:
            # Don't waste space by saving pretrained CBF/Actor
            super(PlanGCBFPlus, self).save(save_dir)
        elif not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save planner
        print(f"Saving planner to {save_dir}")
        pickle.dump(self.planner_train_state.params, open(os.path.join(model_dir, 'planner.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        """Load planner, CBF, and Actor from load_dir/step/. Only CBF and Actor are needed by default."""

        # If planner exists load
        path = os.path.join(load_dir, str(step))
        if os.path.exists(os.path.join(path, 'actor.pkl')):
            print(f"Loading CBF and Actor from {path}")
            super(PlanGCBFPlus, self).load(load_dir, step)

        if os.path.exists(os.path.join(path, 'planner.pkl')):
            print(f"Loading Planner from {path} ")
            self.planner_train_state = \
                self.planner_train_state.replace(params=pickle.load(open(os.path.join(path, 'planner.pkl'), 'rb')))
        else:
            print(f"Planner not found at {path}. Nothing to load.")
