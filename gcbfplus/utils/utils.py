import pathlib
from datetime import timedelta
from typing import Any, Callable, Iterable, ParamSpec, Sequence, TypeVar, List, NamedTuple

import einops as ei
import jax.lax as lax
import matplotlib.collections as mcollections
import numpy as np
from jax import numpy as jnp, tree_util as jtu
from jax._src.lib import xla_client as xc
from matplotlib.animation import FuncAnimation
from rich.progress import Progress, ProgressColumn
from rich.text import Text


def merge01(x):
    return ei.rearrange(x, "n1 n2 ... -> (n1 n2) ...")


_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]

_PyTree = TypeVar("_PyTree")


def jax_vmap(fn: _Fn, in_axes: int | Sequence[Any] = 0, out_axes: Any = 0) -> _Fn:
    return jax.vmap(fn, in_axes, out_axes)


def concat_at_front(arr1: jnp.ndarray, arr2: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    :param arr1: (nx, )
    :param arr2: (T, nx)
    :param axis: Which axis for arr2 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr2_shape = list(arr2.shape)
    del arr2_shape[axis]
    assert np.all(np.array(arr2_shape) == np.array(arr1.shape))

    if isinstance(arr1, np.ndarray):
        return np.concatenate([np.expand_dims(arr1, axis=axis), arr2], axis=axis)
    else:
        return jnp.concatenate([jnp.expand_dims(arr1, axis=axis), arr2], axis=axis)


def tree_concat_at_front(tree1: _PyTree, tree2: _PyTree, axis: int) -> _PyTree:
    def tree_concat_at_front_inner(arr1: jnp.ndarray, arr2: jnp.ndarray):
        return concat_at_front(arr1, arr2, axis=axis)

    return jtu.tree_map(tree_concat_at_front_inner, tree1, tree2)


def tree_index(tree: _PyTree, idx: int) -> _PyTree:
    return jtu.tree_map(lambda x: x[idx], tree)


def jax2np(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, pytree)


def np2jax(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(jnp.array, pytree)


def mask2index(mask: jnp.ndarray, n_true: int) -> jnp.ndarray:
    idx = lax.top_k(mask, n_true)[1]
    return idx


def jax_jit_np(
        fn: _Fn,
        static_argnums: int | Sequence[int] | None = None,
        static_argnames: str | Iterable[str] | None = None,
        donate_argnums: int | Sequence[int] = (),
        device: xc.Device = None,
        *args,
        **kwargs,
) -> _Fn:
    jit_fn = jax.jit(fn, static_argnums, static_argnames, donate_argnums, device, *args, **kwargs)

    def wrapper(*args, **kwargs) -> _R:
        return jax2np(jit_fn(*args, **kwargs))

    return wrapper


def chunk_vmap(fn: _Fn, chunks: int) -> _Fn:
    fn_jit_vmap = jax_jit_np(jax.vmap(fn))

    def wrapper(*args) -> _R:
        args = list(args)
        # 1: Get the batch size.
        batch_size = len(jtu.tree_leaves(args[0])[0])
        chunk_idxs = np.array_split(np.arange(batch_size), chunks)

        out = []
        for idxs in chunk_idxs:
            chunk_input = jtu.tree_map(lambda x: x[idxs], args)
            out.append(fn_jit_vmap(*chunk_input))

        # 2: Concatenate the output.
        out = tree_merge(out)
        return out

    return wrapper


class MutablePatchCollection(mcollections.PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self._paths = None
        self.patches = patches
        mcollections.PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


class CustomTimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        delta = timedelta(seconds=elapsed)
        delta = timedelta(seconds=delta.seconds, milliseconds=round(delta.microseconds // 1000))
        delta_str = str(delta)
        return Text(delta_str, style="progress.elapsed")


def save_anim(ani: FuncAnimation, path: pathlib.Path):
    pbar = Progress(*Progress.get_default_columns(), CustomTimeElapsedColumn())
    pbar.start()
    task = pbar.add_task("Animating", total=ani._save_count)

    def progress_callback(curr_frame: int, total_frames: int):
        pbar.update(task, advance=1)

    ani.save(path, progress_callback=progress_callback)
    pbar.stop()


def tree_merge(data: List[NamedTuple]):
    def body(*x):
        x = list(x)
        if isinstance(x[0], np.ndarray):
            return np.concatenate(x, axis=0)
        else:
            return jnp.concatenate(x, axis=0)

    out = jtu.tree_map(body, *data)
    return out


def tree_stack(trees: list):
    def tree_stack_inner(*arrs):
        arrs = list(arrs)
        if isinstance(arrs[0], np.ndarray):
            return np.stack(arrs, axis=0)
        return np.stack(arrs, axis=0)

    return jtu.tree_map(tree_stack_inner, *trees)


from typing import Any, Callable, Union

import optax

import jax
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT

from typing import Tuple


@jtu.register_pytree_with_keys_class
class OptimizerStateTree(tuple):
    """PyTree node for storing a list of Optax optimizers or optax states"""
    optimizers: Tuple[optax.OptState]

    def __init__(self, optimizers):
        self.optimizers = optimizers

    def tree_flatten_with_keys(self):
        flat_contents = [(key, getattr(self, key)) for key in self.__dict__.keys()]
        aux_data = None
        return flat_contents, aux_data

    def tree_flatten(self):
        children = self.optimizers  # Optax optimizers are already PyTrees
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)

    def copy(self):
        return OptimizerStateTree(self.optimizers)


class MultipleLossTrainState(struct.PyTreeNode):
    """Simple train state for the case with multiple Optax optimizers.

    Example usage::

      >>> import flax.linen as nn
      >>> from flax.training.train_state import TrainState
      >>> import jax, jax.numpy as jnp
      >>> import optax

      >>> x = jnp.ones((1, 2))
      >>> y = jnp.ones((1, 2))
      >>> model = nn.Dense(2)
      >>> variables = model.init(jax.random.key(0), x)
      >>> NUM_OPTIMIZERS = 2
      >>> tx = [optax.adam(1e-3)] * NUM_OPTIMIZERS

      >>> state = TrainState.create(
      ...     apply_fn=model.apply,
      ...     params=variables['params'],
      ...     tx=tx)

      >>> def loss_fn(params, x, y):
      ...   predictions = state.apply_fn({'params': params}, x)
      ...   loss = optax.l2_loss(predictions=predictions, targets=y).mean()
      ...   return loss
      >>> def loss_fn2(params, x, y):
      ...   predictions = state.apply_fn({'params': params}, x)
      ...   loss = optax.l1_loss(predictions=predictions, targets=y).mean()
      ...   return loss
      >>> loss_fn(state.params, x, y)
      Array(3.3514676, dtype=float32)
      >>> lossfns = [loss_fn, loss_fn2]

      >>> grads = [ jax.grad(lossfns[_iopt])(state.params, x, y) for _iopt in range(NUM_OPTIMIZERS) ]
      >>> state = state.apply_gradients(grads=grads)
      >>> loss_fn(state.params, x, y)
      Array(3.343844, dtype=float32)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    :ivar step: Counter starts at 0 and is incremented by every call to ``.apply_gradients()``.
    :ivar apply_fn: Usually set to ``model.apply()``. Kept in this dataclass for convenience to have a shorter params list for the ``train_step()`` function in your training loop.
    :ivar params: The parameters to be updated by ``tx`` and used by ``apply_fn``.
    :ivar tx: An Optax gradient transformation.
    :ivar opt_state: The state for ``tx``.

    """

    step: Union[int, jax.Array]
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: List[optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_states: List[optax.OptState] = struct.field(pytree_node=True)

    def apply_gradients(self, *, grads: dict, **kwargs):
        """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.

        Note that internally this function calls ``.tx.update()`` followed by a call
        to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

        :param grads: Gradients that have the same pytree structure as ``.params``.
                    Dict keys should be integers use to index the optimizers.
                    Using dict to allow alternating updates.
        :param kwargs: Additional dataclass attributes that should be ``.replace()``-ed.
        :return: An updated instance of ``self`` with ``step`` incremented by one, ``params``
                    and ``opt_state`` updated by applying ``grads``, and additional attributes
                    replaced as specified by ``kwargs``.
        """
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads['params']
            params_with_opt = self.params['params']
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        new_opt_states = self.opt_states.copy()

        num_losses = len(self.tx)
        for i, grads_for_opt in grads_with_opt.items():
            if grads_for_opt is None:
                # Skip if no gradients are provided.
                continue

            def get_switch_callable(i):
                """Returns an update operation for the i-th optimizer."""

                def update_op(opt_states, params, grads):
                    tx = self.tx[i]
                    updates, new_opt_state = tx.update(
                        grads, opt_states[i], params)
                    params = optax.apply_updates(params, updates)
                    new_opt_states = opt_states.copy()
                    new_opt_states[i] = new_opt_state
                    return params, new_opt_states

                return update_op

            # Pick the right update and transform
            params_with_opt, new_opt_states = jax.lax.switch(i, [get_switch_callable(_i) for _i in range(num_losses)],
                                                             new_opt_states,
                                                             params_with_opt, grads_for_opt)
            # As implied by the OWG name, the gradients are used directly to update the
        # parameters.
        if OVERWRITE_WITH_GRADIENT in grads:
            # TODO: May not work with multiple losses
            new_params = {
                'params': params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_states=new_opt_states,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx: List, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        params_with_opt = (
            params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        )
        opt_states = []
        for i, _tx in enumerate(tx):
            # Init each opt state
            opt_state = _tx.init(params_with_opt)
            opt_states.append(opt_state)

        # opt_state = tx.init(params_with_opt)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_states=opt_states,
            **kwargs,
        )
