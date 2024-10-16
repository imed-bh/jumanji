import chex


@chex.dataclass
class Operations:
    machine_ids: chex.Array  # (num_jobs, max_num_ops)
    durations: chex.Array  # (num_jobs, max_num_ops)
    mask: chex.Array  # (num_jobs, max_num_ops)
    scheduled_times: chex.Array  # (num_jobs, max_num_ops)
