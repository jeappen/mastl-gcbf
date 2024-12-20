"""Some planner NNs to be used in the planner module"""

from typing import Dict

import flax.linen as nn

from .utils import default_nn_init, scaled_init, AnyFloat, HidSizes, ActFn, signal_last_enumerate


# class GRUNet(nn.Module):
#     def __init__(self, goal_space_dim: int, arch_kwargs: Dict = None):
#         super(GRUNet, self).__init__()
#
#         self.goal_space_dim = goal_space_dim
#         if arch_kwargs is None:
#             self.arch_kwargs = {
#                 "hidden_size": 128,
#                 "num_layers": 1,
#                 "batch_first": True,
#                 "dropout": False,
#                 "bidirectional": True
#             }
#         else:
#             self.arch_kwargs = arch_kwargs
#
#         self.gru = nn.GRU(input_size=self.goal_space_dim, **self.arch_kwargs)
#         self.linear = nn.Linear(self.arch_kwargs["hidden_size"], self.goal_space_dim)
#
#     def forward(self, x: th.Tensor) -> th.Tensor:
#         return self.linear(self.gru(x))


class EulerOdeNet(nn.Module):
    """Euler ODE net to predict the change in the goal given the current goal."""
    goal_space_dim: int
    hid_sizes: HidSizes

    arch_kwargs: Dict = None
    max_difference: float = 10.0
    time_step: float = 1.

    act: ActFn = nn.relu
    act_final: bool = True
    use_layernorm: bool = False
    scale_final: float | None = None
    dropout_rate: float | None = None

    @nn.compact
    def __call__(self, x: AnyFloat, apply_dropout: bool = False) -> AnyFloat:
        """Only gives the change in the goal. The final goal is obtained by adding this to the current goal."""
        nn_init = default_nn_init
        for is_last_layer, ii, hid_size in signal_last_enumerate(self.hid_sizes):
            if is_last_layer and self.scale_final is not None:
                x = nn.Dense(hid_size, kernel_init=scaled_init(nn_init(), self.scale_final))(x)
            else:
                x = nn.Dense(hid_size, kernel_init=nn_init())(x)

            no_activation = is_last_layer and not self.act_final
            if not no_activation:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate, deterministic=not apply_dropout)(x)
                if self.use_layernorm:
                    x = nn.LayerNorm()(x)
                x = self.act(x)

        ode_out = x
        if self.max_difference is not None:
            ode_out = nn.tanh(ode_out) * self.max_difference
        return self.time_step * ode_out
    # def __init__(self, goal_space_dim: int, arch_kwargs: Dict = None, max_difference: float = 10.0):
    #     super(EulerOdeNet, self).__init__()
    #
    #     self.goal_space_dim = goal_space_dim
    #     if arch_kwargs is None:
    #         self.arch_kwargs = {
    #             "hidden_arch": [128, 64]
    #         }
    #     else:
    #         self.arch_kwargs = arch_kwargs
    #
    #     self.activation = nn.ELU()
    #     self.layers = [nn.Linear(self.goal_space_dim, self.arch_kwargs["hidden_arch"][0])]
    #     self.layers.append(self.activation)
    #     for i in range(len(self.arch_kwargs["hidden_arch"]) - 1):
    #         self.layers.append(nn.Linear(self.arch_kwargs["hidden_arch"][i],
    #                                      self.arch_kwargs["hidden_arch"][i + 1]))
    #         self.layers.append(self.activation)
    #     self.layers.append(nn.Linear(self.arch_kwargs["hidden_arch"][-1], self.goal_space_dim))
    #
    #     self.ode = nn.Sequential(*self.layers)
    #     self.max_difference = max_difference
    #     self.time_step = 1.
