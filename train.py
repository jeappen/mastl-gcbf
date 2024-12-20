import argparse
import datetime
import os

import numpy as np
import wandb
import yaml
# Spec related
from jax import config as jax_config

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.stl.utils import TRAINING_CONFIG, PLANNER_CONFIG
from gcbfplus.trainer.trainer import Trainer
from gcbfplus.trainer.utils import is_connected


def train(args):
    print(f"> Running train.py {args}")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    np.random.seed(args.seed)
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"
        jax_config.update("jax_disable_jit", True)  # Disable JIT for good measure

    # Spec related
    # Update params from config
    max_step = TRAINING_CONFIG['max_step']
    planner_params = {}
    batch_size = 256  # Default batch size
    stl_wrapper = None

    if args.algo == 'plangcbf+':
        # Set planner params
        # initialize with default values
        planner_params.update(PLANNER_CONFIG)
        planner_params['planner'] = PLANNER_CONFIG['planner'] if args.planner is None else args.planner
        planner_params['goal_sample_interval'] = args.goal_sample_interval
        planner_params['lr'] = PLANNER_CONFIG['lr'] if args.lr_planner is None else args.lr_planner
        # params['plan_stl_loss_coeff'] = 2 # Edit for hparam search
        max_step = planner_params['goal_sample_interval'] * int(args.spec_len)
        if args.multi_sample:
            assert int(args.spec_len) % int(
                args.plan_length) == 0, "Spec len should be multiple of plan length for multi-sample mode"
            planner_params['plan_length'] = int(args.plan_length)
        else:
            planner_params['plan_length'] = int(args.spec_len)
        planner_params['plan_stl_loss_coeff'] = args.loss_plan_stl_coef
        planner_params['real_stl_loss_coeff'] = args.loss_real_stl_coef
        planner_params['achievable_loss_coeff'] = args.loss_achievable_coef
        planner_params['mb_train_subset_size'] = PLANNER_CONFIG['mb_train_subset_size']
        planner_params['max_step'] = max_step  # To log the max_step in wandb
        planner_params['add_aux_features'] = args.add_aux_features
        print(f"Setting max_step to {max_step} for planner with plan length {planner_params['plan_length']}")
        planner_params['planner_gnn_layers'] = PLANNER_CONFIG['planner_gnn_layers']
        planner_params['skip_actor_update'] = PLANNER_CONFIG['skip_actor_update']
        print(f"Skip actor update: {planner_params['skip_actor_update']}")

        # Find batch size that divides max step
        batch_size = max_step
        # while max_step % batch_size != 0:
        #     batch_size -= 1

        # Make environment with STL wrapper
        from gcbfplus.env.wrapper import NeuralSTLWrapper
        if args.algo == 'plangcbf+':
            stl_wrapper = lambda x: NeuralSTLWrapper(x, spec=args.spec, spec_len=args.spec_len,
                                                     plan_length=planner_params['plan_length'],
                                                     goal_sample_interval=planner_params['goal_sample_interval'],
                                                     max_step=max_step,
                                                     add_aux_features=planner_params['add_aux_features'])
        else:
            print("Not using STL wrapper")

    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        wrapper_fn=stl_wrapper,
        max_step=max_step
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        wrapper_fn=stl_wrapper,
        max_step=max_step
    )

    # create low level controller
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=args.gnn_layers,
        batch_size=batch_size,
        buffer_size=args.buffer_size,
        horizon=args.horizon,
        lr_actor=args.lr_actor,
        lr_cbf=args.lr_cbf,
        alpha=args.alpha,
        eps=0.02,
        inner_epoch=args.inner_epoch,
        loss_action_coef=args.loss_action_coef,
        loss_unsafe_coef=args.loss_unsafe_coef,
        loss_safe_coef=args.loss_safe_coef,
        loss_h_dot_coef=args.loss_h_dot_coef,
        max_grad_norm=2.0,
        seed=args.seed,
        load_dir=args.load_dir,
        params=planner_params
    )

    # set up logger
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%Y%m%d%H%M%S")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(f"{args.log_dir}/{args.env}"):
        os.makedirs(f"{args.log_dir}/{args.env}")
    if not os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}"):
        os.makedirs(f"{args.log_dir}/{args.env}/{args.algo}")
    log_dir = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}"
    run_name = f"{args.algo}_{args.env}_{start_time}" if args.name is None else args.name

    # get training parameters
    train_params = {
        "run_name": run_name,
        "training_steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval,
    }

    # create trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
    )

    # save config
    wandb.config.update(args)
    wandb.config.update(algo.config)
    wandb.config.update(trainer.config)
    if not args.debug:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(args, f)
            yaml.dump(algo.config, f)

    # start training
    trainer.train()


def main():
    parser = argparse.ArgumentParser()

    # custom arguments
    parser.add_argument("-n", "--num-agents", type=int, default=8)
    parser.add_argument("--algo", type=str, default="gcbf+")
    parser.add_argument("--env", type=str, default="SimpleCar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--area-size", type=float, required=True)

    # gcbf / gcbf+ arguments
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-cbf", type=float, default=3e-5)
    parser.add_argument("--loss-action-coef", type=float, default=0.0001)
    parser.add_argument("--loss-unsafe-coef", type=float, default=1.0)
    parser.add_argument("--loss-safe-coef", type=float, default=1.0)
    parser.add_argument("--loss-h-dot-coef", type=float, default=0.01)
    parser.add_argument("--buffer-size", type=int, default=512)

    # default arguments
    parser.add_argument("--n-env-train", type=int, default=16)
    parser.add_argument("--n-env-test", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)

    # Spec and planner related
    parser.add_argument('--spec', type=str, default="seq2")
    parser.add_argument('--spec-len', type=int, default=20)
    parser.add_argument('--load-dir', type=str, default=None, help="Directory to load cbf model from")
    parser.add_argument('--planner', type=str, default="ode", help="If specified, sets the planner to use")
    parser.add_argument("--lr-planner", type=float, default=3e-5)
    parser.add_argument("--loss-plan-stl-coef", type=float, default=0.1)
    parser.add_argument("--loss-real-stl-coef", type=float, default=1.0)
    parser.add_argument("--loss-achievable-coef", type=float, default=0.1)
    parser.add_argument("--inner-epoch", type=int, default=TRAINING_CONFIG['inner_epoch'],
                        help="Number of inner epochs for the algorithm update")
    parser.add_argument("--goal-sample-interval", type=int, default=PLANNER_CONFIG['goal_sample_interval'],
                        help="Number of steps between goal changes")  # set default to work with wandb config
    parser.add_argument("--plan-length", type=int, default=PLANNER_CONFIG['plan_length'],
                        help="Number of steps between running the planner")  # set default to work with wandb config
    parser.add_argument('--add-aux-features', action='store_true', default=False,
                        help="Calculate aux features for planner and add to node features")
    parser.add_argument('--multi-sample', action='store_true', default=False,
                        help="Sample multiple goals in a single episode")

    # Hparam tuning (Not implemented)
    parser.add_argument('--ray', action='store_true', default=False, help="Use ray for hparam tuning")
    parser.add_argument('--num-cpu', type=int, default=32, help="Number of CPUs for ray")
    parser.add_argument('--num-cpu-per-worker', type=int, default=4, help="Number of CPUs per worker for ray")
    parser.add_argument('--test', action='store_true', default=False, help="Run in test mode")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    # with ipdb.launch_ipdb_on_exception():
    main()
