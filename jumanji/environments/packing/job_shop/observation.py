from typing import NamedTuple

import chex


class Observation(NamedTuple):
    """
    ops_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_mask: for each job, indicates which operations remain to be scheduled. False if the
        op has been scheduled or if the op was added for padding, True otherwise. The first True in
        each row (i.e. each job) identifies the next operation for that job.
    machines_job_ids: for each machine, it specifies the job currently being processed. Note that
        the index num_jobs represents a no-op for which the time until available is always 0.
    machines_remaining_times: for each machine, it specifies the number of time steps until
        available.
    action_mask: for each machine, it indicates which jobs (or no-op) can legally be scheduled.
        The last column corresponds to no-op.
    """

    ops_machine_ids: chex.Array  # (num_jobs, max_num_ops)
    ops_durations: chex.Array  # (num_jobs, max_num_ops)
    ops_mask: chex.Array  # (num_jobs, max_num_ops)
    machines_job_ids: chex.Array  # (num_machines,)
    machines_remaining_times: chex.Array  # (num_machines,)
    action_mask: chex.Array  # (num_machines, num_jobs + 1)
