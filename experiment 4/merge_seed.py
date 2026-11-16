"""Seed only the weighted merge, restoring RNG states afterwards."""
from contextlib import contextmanager
import random


@contextmanager
def isolated_merge_seed(seed):
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def install_merge_seed(seed):
    import bench.merging as merging

    original = merging._merge_into_new_adapter

    def seeded_merge(model, adapters, config):
        if config.strategy.startswith('dare_'):
            if seed is None:
                raise ValueError('DARE requires an explicit independent merge seed')
            with isolated_merge_seed(seed):
                return original(model, adapters, config)
        return original(model, adapters, config)

    merging._merge_into_new_adapter = seeded_merge
