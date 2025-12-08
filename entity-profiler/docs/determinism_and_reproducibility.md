# Determinism and Reproducibility

The project enforces deterministic behavior where possible:

- Global RNG seeds are set in `config.set_global_seed`.
- Sorting and ordering operations are explicit and deterministic.
- Background subtraction and other classical CV ops are deterministic.

Some operations (e.g. certain BLAS calls) may still introduce tiny numerical differences across platforms.
