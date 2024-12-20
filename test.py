import argparse
import datetime
import functools as ft
import jax
# Spec related
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import os
import pathlib
import yaml

from gcbfplus.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.env.wrapper import STLWrapper, NeuralSTLWrapper, AsyncNeuralSTLWrapper, AsyncSTLWrapper, FormationWrapper, \
    AsyncFormationWrapper, ASYNC_WRAPPER_LIST
from gcbfplus.stl.utils import PLANNER_CONFIG, STL_INFO_KEYS
from gcbfplus.trainer.utils import get_bb_cbf
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, jax_vmap


def test(args):
    print(f"> Running test.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)  # Disable JIT for good measure
    np.random.seed(args.seed)

    config = {}

    # load config
    if not args.u_ref and args.path is not None:
        print(f"Loading config from {args.path}")
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Spec related
    # Update params from config
    max_step = args.max_step
    load_dir = args.load_dir
    params = {}
    planner = args.planner
    spec = args.spec
    spec_len = args.spec_len
    # Only time preprocess for non-MILP planners and loaded configs
    time_preprocess = not args.u_ref and not isinstance(config, dict) and (config.algo == 'plangcbf+') and (
            planner != 'stlpy') and not args.debug and not args.skip_time_preprocess
    end_when_done = args.async_planner  # End when done for async planner

    stl_wrapper = None
    if config and config.algo == 'plangcbf+':
        # Load config params unless specified
        # Set specific planner params
        for k in PLANNER_CONFIG.keys():
            if k in config:
                print(f"Loading {k} from config {config.__getattribute__(k)}")
                params[k] = config.__getattribute__(k)
        params['planner'] = config.planner
        params['lr'] = config.lr
        planner = params['planner']
        spec_len = int(config.spec_len)
        spec = config.spec
        if args.goal_sample_interval is not None:
            print(f"Overriding config and setting goal_sample_interval to {args.goal_sample_interval} for planner "
                  f"vs {params.get('goal_sample_interval', None)} in config")
            params['goal_sample_interval'] = args.goal_sample_interval
        if args.plan_length is not None:
            print(f"Overriding config and setting plan_length to {args.plan_length} for planner "
                  f"vs {params.get('plan_length', None)} in config")
            params['plan_length'] = args.plan_length
            assert int(args.spec_len) % int(
                args.plan_length) == 0, "Spec len should be multiple of plan length for multi-sample mode"

        if args.spec_len is not None:
            print(f"Overriding config and setting spec_len to {args.spec_len} for planner "
                  f"vs {config.spec_len} in config")
            spec_len = args.spec_len

        if args.spec is not None:
            print(f"Overriding config and setting spec to {args.spec} for planner "
                  f"vs {config.spec} in config")
            spec = args.spec

        if not args.multi_sample:
            params['plan_length'] = spec_len

        max_step_factor = 8 if 'loop' in spec else 5

        max_step = params['goal_sample_interval'] * spec_len * max_step_factor  # Longer max step for leniency
        print(f"Setting max_step to {max_step} for planner")

        # Check if model exists
        # path = args.path
        # model_path = os.path.join(path, "models")
        # if args.step is None:
        #     models = os.listdir(model_path)
        #     step = max([int(model) for model in models if model.isdigit()])
        # else:
        #     step = args.step
        # if not os.path.exists(os.path.join(args.path, "models", str(step), 'actor.pkl')):
        #     print(f"Model not found at {os.path.join(args.path, 'models', str(step), 'actor.pkl')}")
        params['async_reach_radius'] = PLANNER_CONFIG['async_reach_radius']

        if args.async_planner:
            print("Using async planner")
            stl_wrapper = AsyncNeuralSTLWrapper
        else:
            stl_wrapper = NeuralSTLWrapper

        load_dir = config.load_dir
        # Make environment with STL wrapper
        stl_wrapper = ft.partial(stl_wrapper, spec=spec, spec_len=spec_len,
                                 goal_sample_interval=params['goal_sample_interval'],
                                 max_step=max_step, plan_length=params['plan_length'],
                                 async_reach_radius=params['async_reach_radius'])
    elif args.planner in ['stlpy']:
        param_keys = ['async_reach_radius']
        for k in PLANNER_CONFIG.keys():
            if k in args:
                print(f"Loading {k} from args {args.__getattribute__(k)}")
                params[k] = PLANNER_CONFIG[k]
            if k in param_keys:
                print(f"Loading {k} from args {PLANNER_CONFIG[k]}")
                params[k] = PLANNER_CONFIG[k]

        if args.goal_sample_interval is not None:
            print(f"Setting goal_sample_interval to {args.goal_sample_interval} for planner ")
            params['goal_sample_interval'] = args.goal_sample_interval
        assert (args.spec_len is not None), "spec_len must be specified for STL planner"
        assert (args.spec is not None), "spec must be specified for STL planner"
        spec_len = args.spec_len
        max_step_factor = 8 if 'loop' in spec else (1.5 if 'reach' in spec else 5)  # Longer max step for leniency
        max_step = int(params['goal_sample_interval'] * spec_len * max_step_factor)  # Longer max step for leniency
        print(f"Setting max_step to {max_step} for planner "
              f"= goal_sample_interval({params['goal_sample_interval']}) * "
              f"spec_len({spec_len}) * max_step_factor({max_step_factor})")

        if args.async_planner:
            print("Using async planner")
            stl_wrapper = AsyncSTLWrapper
        else:
            stl_wrapper = STLWrapper

        if args.single_plan:
            print("Using single plan mode")
            if args.async_planner:
                stl_wrapper = ft.partial(AsyncFormationWrapper, formation=args.single_plan_formation)
            else:
                stl_wrapper = ft.partial(FormationWrapper, formation=args.single_plan_formation)

        stl_wrapper = ft.partial(stl_wrapper, spec=args.spec, spec_len=args.spec_len, max_step=max_step,
                                 goal_sample_interval=params['goal_sample_interval'],
                                 async_reach_radius=params['async_reach_radius'])
    else:
        print("Not using STL wrapper")

    # create environments
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=max_step,
        max_travel=args.max_travel,
        wrapper_fn=stl_wrapper
    )

    # No need to save graph if no rollout metrics
    nograph = args.no_video and not isinstance(env, STLWrapper)

    if not args.u_ref:
        if args.path is not None:
            path = args.path
            model_path = os.path.join(path, "models")
            if args.step is None:
                models = os.listdir(model_path)
                step = max([int(model) for model in models if model.isdigit()])
            else:
                step = args.step
            print("step: ", step)

            print(f"Load dir: {load_dir}")
            algo = make_algo(
                algo=config.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                gnn_layers=config.gnn_layers,
                batch_size=config.batch_size,
                buffer_size=config.buffer_size,
                horizon=config.horizon,
                lr_actor=config.lr_actor,
                lr_cbf=config.lr_cbf,
                alpha=config.alpha,
                eps=0.02,
                inner_epoch=8,
                loss_action_coef=config.loss_action_coef,
                loss_unsafe_coef=config.loss_unsafe_coef,
                loss_safe_coef=config.loss_safe_coef,
                loss_h_dot_coef=config.loss_h_dot_coef,
                max_grad_norm=2.0,
                seed=config.seed,
                load_dir=load_dir,
                params=params,
                online_pol_refine=args.online_pol_opt,
            )
            algo.load(model_path, step)
            act_fn = jax.jit(algo.act)
            preprocess_graph = jax.jit(algo.preprocess_graph)
        else:
            algo = make_algo(
                algo=args.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                alpha=args.alpha,
            )
            act_fn = jax.jit(algo.act)
            preprocess_graph = jax.jit(algo.preprocess_graph)
            path = os.path.join(f"./logs/{args.env}/{args.algo}")
            if not os.path.exists(path):
                os.makedirs(path)
            step = None
    else:
        assert args.env is not None
        path = os.path.join(f"./logs/{args.env}/nominal")
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
        if not os.path.exists(os.path.join("./logs", args.env)):
            os.mkdir(os.path.join("./logs", args.env))
        if not os.path.exists(path):
            os.mkdir(path)
        algo = None
        preprocess_graph = lambda x: x  # No preprocess for nominal
        act_fn = jax.jit(env.u_ref)
        step = 0

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    algo_is_cbf = isinstance(algo, (CentralizedCBF, DecShareCBF))

    if args.cbf is not None:
        assert isinstance(algo, GCBF) or isinstance(algo, GCBFPlus) or isinstance(algo, CentralizedCBF)
        get_bb_cbf_fn_ = ft.partial(get_bb_cbf, algo.get_cbf, env, agent_id=args.cbf, x_dim=0, y_dim=1)
        get_bb_cbf_fn_ = jax_jit_np(get_bb_cbf_fn_)

        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h
    else:
        get_bb_cbf_fn = None
        cbf_fn = None

    if args.nojit_rollout:
        print("Only jit step, no jit rollout!")
        rollout_fn = env.rollout_fn_jitstep(act_fn, args.max_step, noedge=True, nograph=nograph,
                                            preprocess_graph=preprocess_graph, time_preprocess=time_preprocess,
                                            end_when_done=end_when_done)

        is_unsafe_fn = None
        is_finish_fn = None
    else:
        print("jit rollout!")
        rollout_fn = jax_jit_np(env.rollout_fn(act_fn, args.max_step, preprocess_graph=preprocess_graph))

        is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
        is_finish_fn = jax_jit_np(env.process_finished_rollouts())

    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    is_finishes = []
    rates = []
    cbfs = []
    finish_infos = []
    finish_info_str = ""
    info_metrics = []
    planner_str = planner + ('_uref' if args.u_ref else '')
    planner_str += '_safe' if args.online_pol_opt else ''
    stl_info = [f'{planner_str},{spec_len},{spec},{args.async_planner}']
    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)

        if args.nojit_rollout:
            rollout: RolloutResult
            rollout, is_unsafe, is_finish, rollout_info = rollout_fn(key_x0)
            # if not jnp.isnan(rollout.T_reward).any():
            is_unsafes.append(is_unsafe)
            is_finishes.append(is_finish)
            info_metrics.append(rollout_info)
            print('plan info', rollout_info)
        else:
            raise NotImplementedError("TODO: Implement for STL")
            rollout: RolloutResult = rollout_fn(key_x0)
            is_unsafes.append(is_unsafe_fn(rollout.Tp1_graph))
            is_finishes.append(is_finish_fn(rollout.Tp1_graph))
        if isinstance(env, STLWrapper):
            # Get STL related satisfaction metrics
            # if env is instance of any class in ASYNC_WRAPPER_LIST, then process the finished rollouts info
            if any(isinstance(env, cls) for cls in ASYNC_WRAPPER_LIST):
                finish_metrics = env.process_finished_rollouts_info()(rollout.Tp1_graph, rollout.T_info['changed_goal'])
            else:
                finish_metrics = env.process_finished_rollouts_info()(rollout.Tp1_graph)
            # Overwrite is_finishes with STL related satisfaction
            is_finishes[-1] = jnp.array([finish_metrics[STL_INFO_KEYS[-2]]])  # Shape appropriately
            finish_info = {f"eval/{k}": jnp.nanmean(v) for k, v in finish_metrics.items()}
            finish_info_str = ', ' + ', '.join([f'{k}: {jnp.nanmean(v) :4.2f}' for k, v in finish_metrics.items()])

            if finish_info:
                finish_infos.append(finish_info)

            # Vary the safety calculation depending on the mode (also change the is_unsafes to reflect different ep lengths)
            safe_rate, finish_rate, success_rate, is_unsafes[-1] = env.calc_safety_success_finish(is_unsafes[-1],
                                                                                                  is_finishes[-1],
                                                                                                  rollout,
                                                                                                  ignore_on_finish=args.ignore_on_finish)
        else:
            # Regular finish metrics
            safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
            finish_rate = is_finishes[-1].max(axis=0).mean()
            success_rate = ((1 - is_unsafes[-1].max(axis=0)) * is_finishes[-1].max(axis=0)).mean()
        epi_reward = rollout.T_reward.sum()
        epi_cost = rollout.T_cost.sum()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)

        if args.cbf is not None:
            cbfs.append(get_bb_cbf_fn(rollout.Tp1_graph))
        else:
            cbfs.append(None)
        if len(is_unsafes) == 0:
            continue
        print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, "
              f"safe rate: {safe_rate * 100:.3f}%,"
              f"finish rate: {finish_rate * 100:.3f}%, "
              f"success rate: {success_rate * 100:.3f}%"
              f"{finish_info_str}")

        rates.append(np.array([safe_rate, finish_rate, success_rate]))
    # is_unsafes may have different lengths, stack them to get the max over all
    is_unsafe = np.stack(list(map(lambda x: x.max(axis=0), is_unsafes)))
    is_finish = np.max(np.stack(is_finishes), axis=1)

    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    finish_mean, finish_std = is_finish.mean(), is_finish.std()
    # Note: Does not consider ignore_on_finish here
    success_mean, success_std = ((1 - is_unsafe) * is_finish).mean(), ((1 - is_unsafe) * is_finish).std()

    # Get the mean of a list of dictionaries finish_metrics if they exist
    if finish_infos:
        final_finish_metrics_mean = {f"{k}": jnp.nanmean(jnp.stack([v[k] for v in finish_infos])) for k in
                                     finish_infos[0].keys()}
        final_finish_metrics_std = {f"{k}_std": jnp.nanstd(jnp.stack([v[k] for v in finish_infos])) for k in
                                    finish_infos[0].keys()}
        final_finish_metrics = {**final_finish_metrics_mean, **final_finish_metrics_std}
        finish_info_str = ', Mean scores: ' + ', '.join(
            [f'{k}: {v :4.2f}' for k, v in final_finish_metrics_mean.items()])

    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%, "
        f"finish_rate: {finish_mean * 100:.3f}%, "
        f"success_rate: {success_mean * 100:.3f}%"
        f"{finish_info_str}"
    )

    # save results
    if args.log:
        # calculate mean and std of keys of list of dicts info_metrics using numpy
        info_metrics_mean = {f"{k}_mean": jnp.nanmean(jnp.stack([v[k] for v in info_metrics])) for k in
                             info_metrics[0].keys()}
        info_metrics_std = {f"{k}_std": jnp.nanstd(jnp.stack([v[k] for v in info_metrics])) for k in
                            info_metrics[0].keys()}
        info_metrics = {**info_metrics_mean, **info_metrics_std}
        # consistent ordering of keys and vals
        info_keys_and_vals = list(info_metrics.items())
        info_keys = [k for k, v in info_keys_and_vals]
        info_vals = [v for k, v in info_keys_and_vals]

        header = ["num_agents", "epi", "max_step", "area_size", "n_obs", "safe_mean", "safe_std", "finish_mean",
                  "finish_std", "success_mean", "success_std"] + STL_INFO_KEYS + [f"{k}_std" for k in STL_INFO_KEYS] + [
                     "planner", "spec_len", "spec", "async_planner"] + info_keys
        if not os.path.exists(os.path.join(path, "test_log.csv")):
            # write header
            with open(os.path.join(path, "test_log.csv"), "w") as f:
                f.write(','.join(header) + '\n')
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            info_to_write = [f'{env.num_agents}', f'{args.epi}', f'{env.max_episode_steps}',
                             f'{env.area_size}', f'{env.params["n_obs"]}', f'{safe_mean * 100:.3f}',
                             f'{safe_std * 100:.3f}', f'{finish_mean * 100:.3f}', f'{finish_std * 100:.3f}',
                             f'{success_mean * 100:.3f}', f'{success_std * 100:.3f}']
            stl_info_metrics = [f'{v :4.2f}' for k, v in final_finish_metrics.items()] + stl_info + [
                f'{v :4.4f}' for v in info_vals]
            f.write(','.join(info_to_write + stl_info_metrics) + '\n')

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe, cbf) in enumerate(zip(rollouts, is_unsafes, cbfs)):
        if algo_is_cbf:
            safe_rate, finish_rate, success_rate = rates[ii] * 100
            video_name = f"n{num_agents}_epi{ii:02}_sr{safe_rate:.0f}_fr{finish_rate:.0f}_sr{success_rate:.0f}"
        else:
            video_name = f"n{num_agents}_step{step}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}"
        video_name += f"{','.join(stl_info)}"

        viz_opts = {}
        if args.cbf is not None:
            video_name += f"_cbf{args.cbf}"
            viz_opts["cbf"] = [*cbf, args.cbf]
        if args.plot_async_change and args.async_planner:
            viz_opts['plot_changed_goal'] = True
            if args.plot_snapshot:
                viz_opts['plot_snapshot'] = True
                if args.plot_pgf:
                    viz_opts['plot_pgf'] = True

        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--area-size", type=float, required=True)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--cbf", type=int, default=None)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--u-ref", action="store_true", default=False)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)

    # Spec and planner related
    parser.add_argument('--load-dir', type=str, default=None,
                        help="Directory to load actor/cbf model from for plangcbf+")
    parser.add_argument('--spec', type=str, default=None)  # Use loaded config unless specified
    parser.add_argument('--spec-len', type=int, default=None)
    parser.add_argument("--goal-sample-interval", type=int, default=None,
                        help="Number of steps between sampling the planner")
    parser.add_argument('--planner', type=str, default=None, help="If specified, sets the planner to use",
                        choices=['stlpy', 'mamps'])
    parser.add_argument('--plan-length', type=int, default=None, help="Number of steps to plan for")
    parser.add_argument("--async-planner", action="store_true", default=False, help="Use async planner")
    parser.add_argument("--skip-time-preprocess", action="store_true", default=False,
                        help="Skip time preprocess for planner")
    parser.add_argument('--multi-sample', action='store_true', default=False,
                        help="Sample multiple goals in a single episode")
    parser.add_argument('--ignore-on-finish', action='store_true', default=False,
                        help="Ignore safety on finish for STL")

    # Ablation options
    parser.add_argument("--single-plan", action="store_true", default=False,
                        help="Single plan mode")  # All agents share the same plan
    parser.add_argument("--single-plan-formation", type=str, default=None, help="Single plan formation",
                        choices=[None, 'line'])  # Formation for single plan mode

    # Extra debug options
    parser.add_argument('--online-pol-opt', action='store_true', default=False,
                        help="Online policy optimization for GCBF+, prioritize safety")
    parser.add_argument("--plot-async-change", action="store_true", default=False, help="Plot async change points")
    parser.add_argument("--plot-snapshot", action="store_true", default=False, help="Plot trajectory snapshot")
    parser.add_argument("--plot-pgf", action="store_true", default=False, help="Plot trajectory snapshot in pgf")

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    # with ipdb.launch_ipdb_on_exception():
    main()
