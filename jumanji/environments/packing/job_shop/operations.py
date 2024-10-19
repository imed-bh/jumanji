from __future__ import annotations
import jax
import jax.numpy as jnp
import chex


@chex.dataclass
class Operations:
    machine_ids: chex.Array  # (num_jobs, max_num_ops)
    durations: chex.Array  # (num_jobs, max_num_ops)
    mask: chex.Array  # (num_jobs, max_num_ops)
    scheduled_times: chex.Array  # (num_jobs, max_num_ops)

    @property
    def num_jobs(self):
        return self.machine_ids.shape[0]

    @property
    def max_num_ops(self):
        return self.machine_ids.shape[1]

    @property
    def next_op_ids(self):
        return jnp.argmax(self.mask, axis=-1)

    def next_op_durations(self, job_ids):
        op_ids = self.next_op_ids[job_ids]
        return self.durations[job_ids, op_ids]

    def is_job_finished(self, job_id):
        return jnp.all(~self.mask[job_id])

    def next_machine_id_for_job(self, job_id):
        op_id = self.next_op_ids[job_id]
        return self.machine_ids[job_id, op_id]

    def update(self, action: chex.Array, step_count: chex.Array) -> Operations:
        job_ids = jnp.arange(self.num_jobs)

        is_new_job = jnp.isin(job_ids, action)
        is_new_job = jnp.expand_dims(is_new_job, axis=-1)

        is_next_op = jnp.zeros(shape=(self.num_jobs, self.max_num_ops), dtype=bool)
        is_next_op = is_next_op.at[job_ids, self.next_op_ids].set(True)

        is_new_op = is_new_job & is_next_op
        mask = self.mask & ~is_new_op
        scheduled_times = jnp.where(is_new_op, step_count, self.scheduled_times)

        return Operations(
            machine_ids=self.machine_ids,
            durations=self.durations,
            mask=mask,
            scheduled_times=scheduled_times
        )


def operations_flatten(operations: Operations):
    children = (
        operations.machine_ids,
        operations.durations,
        operations.mask,
        operations.scheduled_times
    )
    return children, None


def operations_unflatten(aux_data, children):
    machine_ids, durations, mask, scheduled_times = children
    return Operations(
        machine_ids=machine_ids,
        durations=durations,
        mask=mask,
        scheduled_times=scheduled_times
    )


jax.tree_util.register_pytree_node(
    Operations,
    operations_flatten,
    operations_unflatten
)