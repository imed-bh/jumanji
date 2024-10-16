import jax
import jax.numpy as jnp
import chex


@chex.dataclass
class Machines:
    job_ids: chex.Array  # (num_machines,)
    remaining_times: chex.Array  # (num_machines,)

    @property
    def num_machines(self):
        return self.job_ids.shape[0]

    def is_machine_available(self, machine_id):
        return self.remaining_times[machine_id] == 0

    def is_job_in_progress(self, job_id):
        return jnp.any((self.job_ids == job_id) & (self.remaining_times > 0))


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