import chex


@chex.dataclass
class Machines:
    job_ids: chex.Array  # (num_machines,)
    remaining_times: chex.Array  # (num_machines,)
