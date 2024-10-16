from typing import Any

import jax
import jax.numpy as jnp
import chex

from jumanji.environments.packing.job_shop.machines import Machines
from jumanji.environments.packing.job_shop.operations import Operations


def create_action_mask(
        machines: Machines,
        operations: Operations
) -> chex.Array:

    job_indexes = jnp.arange(operations.num_jobs)
    machine_indexes = jnp.arange(machines.num_machines)

    action_mask = jax.vmap(
        jax.vmap(
            is_action_valid, in_axes=(0, None, None, None)
        ),
        in_axes=(None, 0, None, None)
    )(
        job_indexes,
        machine_indexes,
        machines,
        operations
    )

    full_action_mask = jnp.pad(action_mask, ((0, 0), (0, 1)), constant_values=True)

    return full_action_mask


def is_action_valid(
        job_id: jnp.int32,
        machine_id: jnp.int32,
        machines: Machines,
        operations: Operations,
) -> Any:
    """Check whether a particular action is valid, specifically the action of scheduling
     the specified operation of the specified job on the specified machine given the
     current status of all machines.

     To achieve this, four things need to be checked:
        - The machine is available.
        - The machine is exactly the one required by the operation.
        - The job is not currently being processed on any other machine.
        - The job has not yet finished all of its operations.
    """
    is_machine_available = machines.is_machine_available(machine_id)
    is_correct_machine = operations.next_machine_id_for_job(job_id) == machine_id
    is_job_ready = ~machines.is_job_in_progress(job_id)
    is_job_finished = operations.is_job_finished(job_id)
    return (
            is_machine_available & is_correct_machine & is_job_ready & ~is_job_finished
    )


