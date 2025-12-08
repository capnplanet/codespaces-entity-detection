from contextlib import contextmanager

from ..config import set_global_seed, DEFAULT_SEED


@contextmanager
def deterministic_context(seed: int = DEFAULT_SEED):
    """Context manager to execute a block with a fixed RNG seed."""
    set_global_seed(seed)
    yield
