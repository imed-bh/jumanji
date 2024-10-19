from __future__ import annotations
import jax
import jax.numpy as jnp
import chex

from jumanji.environments.packing.job_shop.arrays import decrement
from jumanji.environments.packing.job_shop.operations import Operations


@chex.dataclass
class Machines:
    job_ids: chex.Array  # (num_machines,)
    remaining_times: chex.Array  # (num_machines,)

    @property
    def num_machines(self):
        return self.job_ids.shape[0]

    def is_available(self):
        return self.remaining_times == 0

    def is_machine_available(self, machine_id):
        return self.remaining_times[machine_id] == 0

    def is_job_in_progress(self, job_id):
        return jnp.any((self.job_ids == job_id) & (self.remaining_times > 0))

    def update(self, action: chex.Array, next_op_durations: chex.Array, noop_index: int) -> Machines:
        noop_actions = action == noop_index

        existing_job_ids = jnp.where(self.is_available(), noop_index, self.job_ids)
        job_ids = jnp.where(noop_actions, existing_job_ids, action)

        remaining_times = jnp.where(noop_actions, self.remaining_times, next_op_durations)
        remaining_times = decrement(remaining_times)

        return Machines(
            job_ids=job_ids,
            remaining_times=remaining_times
        )

def machines_flatten(machines: Machines):
    children = (
        machines.job_ids,
        machines.remaining_times
    )
    return children, None


def machines_unflatten(aux_data, children):
    job_ids, remaining_times = children
    return Machines(
        job_ids=job_ids,
        remaining_times=remaining_times
    )


jax.tree_util.register_pytree_node(
    Machines,
    machines_flatten,
    machines_unflatten
)