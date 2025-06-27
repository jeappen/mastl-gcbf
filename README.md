# MASTL-GCBF

![jax_badge][jax_badge_link]

Official Implementation of Paper: [Joe Eappen](https://jeappen.com), [Zikang Xiong](https://xiong.zikang.me/), [Dipam Patel](https://www.dipampatel.in/), [Aniket Bera](https://www.cs.purdue.edu/homes/ab/),  [Suresh Jagannathan](https://www.cs.purdue.edu/homes/suresh/): "[Scaling Safe Multi-Agent Control for Signal Temporal Logic Specifications](https://mit-realm.github.io/gcbfplus-website/](https://www.jeappen.com/mastl-gcbf-website/)".

## Main File Structure
```
├── requirements.txt                    # requirements file installed via pip
├── train.py                            # training script
├── test.py                             # testing script
├── plot.ipynb                          # plotting helper notebook
├── *.sh                                # bash scripts for running experiments
├── tests                               # (dir) simple testing scripts to load environment
├── pretrained                          # (dir) pre-trained models (saved GCBF+ model used in the paper)
└── gcbfplus                            # (dir) GCBF+ code from MIT-REALM
    ├── algo                            # (dir) GCBF+ and GCBF code
    │   ├── module                      # (dir) High-level network modules
    │   │   ├── planner                 # (dir) Planner modules    
    │   │   └── ...  
    │   ├── plan_gcbf_plus.py           # Planner with GCBF+ controller algorithm
    │   └── ...                         
    ├── env                             # (dir) Environment code
    │   ├── wrapper                     # (dir) Environment wrappers with STL interface
    │   │   ├── wrapper.py              # Environment wrapper with STL interface
    │   │   ├── async_goal.py           # Asynchronous goal wrapper (for asynchronous goal change during deployment)
    │   │   └── ...
    │   └── ...
    ├── nn                              # (dir) Core Neural network modules  
    ├── stl                             # (dir) Signal Temporal Logic (STL) utilities
    ├── trainer                         # (dir) Training utilities
    └── utils                           # (dir) Utility functions
        ├── configs/default_config.yaml # Default configuration file
        └── ...
```


## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n mastl-gcbf python=3.10
conda activate mastl-gcbf
cd mastl-gcbf
```

Then install jax following the [official instructions](https://github.com/google/jax#installation), and the CPU version of pytorch 
(for easy compatibility with the diff-spec package requirements without messing up the jax installation):

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

and then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

## Installation

Install the package by running: 

```bash
pip install -e .
```

## Run

### Environments

We use the 2D environments from the original GCBF+ [1] paper `SingleIntegrator`, `DoubleIntegrator`, and `DubinsCar`.

### Planners

We provide planners including STLPY MILP planner [2] (`stlpy`), GNN-ODE planner (`gnn-ode`), and ODE planner (`ode`) without the GNN component. Use `--planner` to specify the planner.

### STL Specifications

Experiment with different STL specifications by changing the `--spec` flag. We provide the following STL specifications:
- `coverN`: Cover N regions, e.g., `cover3` covers 3 regions
- `seqN`: Sequence of N regions, e.g., `seq3` sequentially visits 3 regions
- `MbranchN`: M-branch with N regions, e.g., `2branch3` has 2 branches with 3 regions each
- `MloopN`: Loop M times over N regions, e.g., `2loop3` has 2 loops with 3 regions each

### Controllers

For the STLPY MILP [2] controller, use the vanilla GCBF+ controller (`gcbf+`) which does not need to be trained, and for the GNN-ODE planner, use the pretrained GCBF+ controller with the learnable planner (`plangcbf+`). Use `--algo` to specify the controller.

### Hyper-parameters

To reproduce the results shown in our paper, one can refer to [`settings.yaml`](./settings.yaml).

### Train

To train the planner (for the `plangcbf+` setting with a `GNN-ODE` or `ODE` planner) given the pretrained GCBF+ controller use:

```bash
python train.py --algo plangcbf+ --env DubinsCar -n 8 --area-size 4 --n-env-train 8 --n-env-test 8  --load-dir ./pretrained/DubinsCar/gcbf+/models/ --spec cover3 --spec-len 15 --lr-planner 1e-5 --planner gnn-ode --goal-sample-interval 30 --loss-real-stl-coef 0.5 --loss-plan-stl-coef 0.5 --steps 2500 --loss-achievable-coef 10 
```

In our paper, we use 8 agents with 1000 training steps. The training logs will be saved in folder `./logs/<env>/<algo>/seed<seed>_<training-start-time>`. We also provide the following flags:

- `-n`: number of agents
- `--env`: environment, including `SingleIntegrator`, `DoubleIntegrator`, `DubinsCar`, `LinearDrone`, and `CrazyFlie`
- `--algo`: algorithm, including `gcbf`, `gcbf+`
- `--seed`: random seed
- `--steps`: number of training steps
- `--name`: name of the experiment
- `--debug`: debug mode: no recording, no saving, and no JIT
- `--obs`: number of obstacles
- `--n-rays`: number of LiDAR rays
- `--area-size`: side length of the environment
- `--n-env-train`: number of environments for training
- `--n-env-test`: number of environments for testing
- `--log-dir`: path to save the training logs
- `--eval-interval`: interval of evaluation
- `--eval-epi`: number of episodes for evaluation
- `--save-interval`: interval of saving the model
- `--goal-sample-interval`: interval of sampling new goals
- `--spec-len`: length of the STL specification (number of waypoints from the planner)
- `--spec`: STL specification
- `--lr-planner`: learning rate of the planner
- `--planner`: planner, including `gnn-ode`, and `ode`'

In addition to the hyper parameters of [GCBF+](https://github.com/MIT-REALM/gcbfplus/), we use the following flags to specify the hyper-parameters:

- `--lr-planner`: learning rate of the planner
- `--loss-plan-stl-coef`: coefficient of the planned path STL loss
- `--loss-achievable-coef`: coefficient of the achievable STL loss (difference between the planned path and the real path)
- `--loss-real-stl-coef`: (optional) coefficient of the real path STL loss (try differentiating through the environment)
- `--buffer-size`: size of the replay buffer

### Test

To test the learned planner with the spec trained upon, where `log_path` is a path to the log folder (e.g. `logs/DubinsCar/plangcbf+/seed0_20240811003419/`), use:

```bash
python test.py --path <log_path> --epi 1 --area-size 4 -n 2 --obs 0 --nojit-rollout --goal-sample-interval 20 --log --async-planner --ignore-on-finish
```
To use the MILP planner, use `--planner stlpy` as below using the pre-trained GCBF+ controller:

```bash
python test.py --path pretrained/DubinsCar/gcbf+/ --epi 1 --area-size 4 -n 2 --obs 0 --nojit-rollout --planner stlpy --spec-len 15 --goal-sample-interval 20 --spec cover3 --log --async-planner --ignore-on-finish
```


This should report the safety rate, goal reaching rate, and success rate of the learned model, and generate videos of the learned model in `<path-to-log>/videos`. Use the following flags to customize the test:

- `-n`: number of agents
- `--obs`: number of obstacles
- `--area-size`: side length of the environment
- `--max-step`: maximum number of steps for each episode, increase this if you have a large environment
- `--path`: path to the log folder
- `--n-rays`: number of LiDAR rays
- `--alpha`: CBF alpha, used in centralized CBF-QP and decentralized CBF-QP
- `--max-travel`: maximum travel distance of agents
- `--cbf`: plot the CBF contour of this agent, only support 2D environments
- `--seed`: random seed
- `--debug`: debug mode
- `--cpu`: use CPU
- `--env`: test environment (not needed if the log folder is specified using `--path`)
- `--algo`: test algorithm (not needed if the log folder is specified using `--path`)
- `--step`: test step (not needed if testing the last saved model)
- `--epi`: number of episodes to test
- `--offset`: offset of the random seeds
- `--no-video`: do not generate videos
- `--log`: log the results to a file
- `--nojit-rollout`: do not use jit to speed up the rollout, used for large-scale tests
- `--async-planner`: asynchronous goal change during deployment (since it is hard to synchronize an unknown number of agents)
- `--ignore-on-finish`: ignore collisions after reaching the goal (assume agent vanishes/lands)
- `--planner`: (for stlpy) test planner (not needed if the log folder is specified using `--path`)
- `--spec-len`: (for stlpy) length of the STL specification (number of waypoints from the planner)
- `--spec`: (for stlpy) STL specification
### Pre-trained models

We provide the pre-trained GCBF+ controller from [1] in the folder [`pretrained`](pretrained).

## Acknowledgement

This uses an underlying [GCBF+](https://mit-realm.github.io/gcbfplus-website/) [1] controller, and we thank the authors for their 
excellent [implementation](https://github.com/MIT-REALM/gcbfplus/) upon which we added planning capabilities.

## References

[1] [GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control](https://github.com/MIT-REALM/gcbfplus/), Zhang, S. et al.

[2] [Mixed-Integer Programming for Signal Temporal Logic with Fewer Binary Variables](https://github.com/vincekurtz/stlpy), Kurtz, Vincent, & Lin, Hai

[jax_badge_link]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC
