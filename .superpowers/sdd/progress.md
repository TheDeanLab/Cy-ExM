# Fine-grained parallel registration progress

Branch base: `493ecfcbc5e56e4385faa3a35e66aab8d7a3f975`
Plan: `docs/superpowers/plans/2026-07-22-fine-grained-parallel-registration.md`

Baseline:
- Unit: 2043 passed
- Integration: 55 passed

Tasks:
- Task 1: complete (`493ecfc..43a157e`, review clean; final-review minor:
  strengthen the scientific-identity regression through `build_workflow_plan`)
- Task 2: complete (`43a157e..b330439`, review clean)
- Task 3: complete (`b330439..a912b9d`, review clean after two fix rounds)
- Task 4: complete (`a912b9d..879d279`, review findings fixed)
- Task 5: complete through first remediation (`879d279..df85231`); second
  re-review findings remediated in `7ed5b47`; final dependency-chain correction
  committed and verified in `35280c3`; final re-review clean
- Task 6: complete (`35280c3..4a5b31d`, review clean; final-review minor:
  directly pin whole-image expansion, cancellation, and reverse-order multiple
  failures)
- Task 7: complete (`4a5b31d..fc89820`; fake, real-process, scientific,
  reuse, cancellation, and lifecycle gates clean); first independent-review
  findings remediated in `58bd0a9` with exact distributed evidence validation,
  complete scientific-request comparison, and controller-edge regressions;
  Task 9 must reconcile the representative fine-task log with the current coarse
  workflow-result validator
- Task 8: implementation committed in `cca67a3`; independent-review findings
  remediated in `7ff6ccf` with exact dynamic probe identity, typed
  pre-controller terminal causes, pre-record placement admission, explicit
  rejected-placement evidence, numeric >9999 ordering, scheduler-neutral graph
  validation, and post-activation scheduler identity checks; final re-review
  remediation records causally exact partial scale failures, shares authoritative
  controller expansion policy, and retains every retry attempt with strict
  placement admission history in `e9df695`; portable unit/integration and HPC
  collect-only gates clean; final malformed raw-candidate versus validated-worker
  identity remediation committed in `437f37d`
- Task 9: implemented from `437f37d` in `c7acb1a`; independent review found
  pre-controller, full-summary, cancellation-order, SLURM-v2, YAML-identity, and
  genuine ArtifactStore reuse gaps. All findings are remediated locally with one
  canonical Task 6 aggregator, exact reference/terminal-summary evidence,
  production Sequential/Local-Dask/SLURM empty-graph causes, pre-serialization
  partial-summary validation, SLURM granular/placement binding, YAML-built
  identity, and canonical fine-to-coarse local-warp reuse. Full gates: 2,258 unit,
  58 integration, repository Ruff/BasedPyright/pre-commit clean; final remediation
  re-review approved and committed in `9f816c2`. A final adversarial audit then
  found authenticate-before-persist ordering and omitted full SLURM preparation
  descriptor/successor binding gaps. The second remediation keeps new partial
  summaries provisional until evidence validates, persists canonical complete
  version-2 task results, decodes them through `FineTaskResult.from_dict` plus the
  shared validator, binds exact controller tasks, and shares producer-successor
  expansion validation. Full gates are clean (2,265 unit, 58 integration,
  repository Ruff/BasedPyright/pre-commit), and final independent re-review
  approved; the second remediation is committed.
- Task 10: implementation complete pending independent review; bounded real-ANTs
  coarse/fine and fine sequential/Local-Dask parity, unequal XYZ shapes, four
  nonlinear chunks, three moving channels, completion-order-independent legacy
  assembly, and two-stage exact restart are green. A RED-exposed prepared regional
  pixeltype defect is fixed with format-version-2 manifest/request identity and
  exact ANTs reconstruction; full gates: 2,266 unit, 61 integration, repository
  Ruff/BasedPyright/pre-commit clean.
- Task 11: pending
- Task 12: pending
- Task 13: pending
