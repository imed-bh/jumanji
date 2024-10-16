import jax.numpy as jnp
import jax.random
import pytest
from chex import PRNGKey

from jumanji.environments.packing.job_shop.env import JobShop
from jumanji.environments.packing.job_shop.generator import Generator
from jumanji.environments.packing.job_shop.machines import Machines
from jumanji.environments.packing.job_shop.operations import Operations
from jumanji.environments.packing.job_shop.state import State


class DummyGenerator(Generator):
    """Hardcoded `Generator` mainly used for testing and debugging. It deterministically
    outputs a hardcoded instance with 3 jobs, 3 machines, a max of 3 ops for any job, and a max
    duration of 4 time steps for any operation.
    """

    def __init__(self) -> None:
        super().__init__(
            num_jobs=3,
            num_machines=3,
            max_num_ops=3,
            max_op_duration=4,
        )

    def __call__(self, key: PRNGKey) -> State:
        """Call method responsible for generating a new state. It returns a job shop scheduling
        instance without any scheduled jobs.

        Args:
            key: jax random key for any stochasticity used in the generation process. Not used
                in this instance generator.

        Returns:
            A JobShop State.
        """
        del key

        ops_machine_ids = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, -1],
            ],
            jnp.int32,
        )
        ops_durations = jnp.array(
            [
                [3, 2, 2],
                [2, 1, 4],
                [4, 3, -1],
            ],
            jnp.int32,
        )

        # Initially, all machines are available (the index self.num_jobs corresponds to no-op)
        machines_job_ids = jnp.array(
            [self.num_jobs, self.num_jobs, self.num_jobs], jnp.int32
        )
        machines_remaining_times = jnp.array([0, 0, 0], jnp.int32)

        # Initial action mask given the problem instance
        action_mask = jnp.array(
            [
                [True, True, False, True],  # Machine 0 legal actions: Job0/Job1/No-op
                [False, False, True, True],  # Machine 1 legal actions: Job2/No-op
                [False, False, False, True],  # Machine 2 legal actions: No-op
            ],
            bool,
        )

        # Initially, all ops have yet to be scheduled (ignore the padded element)
        ops_mask = jnp.array(
            [[True, True, True], [True, True, True], [True, True, False]], bool
        )

        # Initially, none of the operations have been scheduled
        scheduled_times = jnp.array(
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            jnp.int32,
        )

        step_count = jnp.array(0, jnp.int32)

        state = State(
            operations=Operations(
                machine_ids=ops_machine_ids,
                durations=ops_durations,
                mask=ops_mask,
                scheduled_times=scheduled_times,
            ),
            machines=Machines(
                job_ids=machines_job_ids,
                remaining_times=machines_remaining_times,
            ),
            action_mask=action_mask,
            step_count=step_count,
            key=jax.random.PRNGKey(0),
        )

        return state


@pytest.fixture
def job_shop_env() -> JobShop:
    generator = DummyGenerator()
    return JobShop(generator)
