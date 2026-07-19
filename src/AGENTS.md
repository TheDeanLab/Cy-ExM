# Source-tree guidance

- Package all importable code under `src/cyexm`; do not add top-level compatibility
  packages.
- Dependency direction is `cyexm.registration -> cyexm.io/runtime/metrics`.
  `cyexm.io`, `cyexm.runtime`, and `cyexm.metrics` must not import registration.
- Importing a module must not mutate the environment, configure logging, inspect the
  filesystem for datasets, initialize a scheduler, or start registration work.
- Defer ANTs, NumPy, and SciPy imports until native-thread settings have been resolved
  at the operation boundary. In particular, `cyexm.runtime` must never import those
  scientific libraries.
- Do not import ClearEx at runtime. Extract only the focused behavior approved by the
  plan and record provenance instead of creating cross-repository coupling.
