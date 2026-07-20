# Real-library integration-test guidance

- These tests exercise real ANTSpyX, NumPy, SciPy, and tifffile behavior on small,
  deterministic synthetic volumes. Keep the complete portable integration suite
  below five minutes on one CPU allocation.
- Do not import NumPy, ANTs, SciPy, or tifffile at test-module scope. The autouse
  fixture in `conftest.py` must resolve an explicit `NativeThreadConfig(1, 1)` before
  any scientific runtime is imported.
- Mark every test module that exercises scientific libraries with
  `pytestmark = pytest.mark.integration`. Never skip portable checks by hostname or
  by probing lab storage.
- Limit ordinary registration fixtures to at most `16 x 32 x 32` voxels and use the
  `dry run` accuracy preset. Larger data, representative lab inputs, and access to an
  external scheduler or cluster service require the explicit `hpc` marker.
- Assert geometry, component order, masks, axes, dtype, artifact records, and pure
  chunk geometry exactly. Numerical registration comparisons use only the documented
  `rtol=1e-4`, `atol=1e-5` migration tolerance; explain any different tolerance next
  to its assertion.
- Failure injection may replace only the expensive stage boundary under test. Keep
  artifact persistence, reconstruction, and the remaining ANTs operations real, and
  prove restart from the committed upstream record.
- `legacy_baseline.py` is executed only from the pinned ClearEx environment at commit
  `2311a873b783477add7391ff00245f76490147f0`. It may write only to its explicit
  `--output` directory and must never modify the ClearEx checkout.
- `compare_legacy.py` runs from the Cy-ExM environment. Cross-environment scripts
  must emit machine-readable evidence and exit nonzero on every failed comparison;
  never replace missing evidence with a claimed pass.
- Temporary transforms and images belong under pytest `tmp_path` or an explicit
  caller-provided comparison directory. Do not access or alter the separate spheroid
  registration run or its outputs.
- After an approved scheduler dependency is declared, deterministic bounded
  in-process or local-scheduler checks may run as portable integration tests. This
  bounded local-scheduler coverage must use synthetic tasks. Actual SLURM submission
  remains `hpc` and requires an explicit cluster allocation, cleanup assertions, and
  machine-readable job evidence.
