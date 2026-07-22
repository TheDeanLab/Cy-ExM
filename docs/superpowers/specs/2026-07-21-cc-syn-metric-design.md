# Configurable CC SyN Metric Design

**Status:** Approved conversational design pending written-spec review

**Date:** 2026-07-21

## Context

Cy-ExM currently fixes nonlinear registration to Mattes mutual information with 32
bins. The policy is embedded independently in the ANTSpyX call and in artifact
request evidence. Scientific inspection of a completed spheroid registration found
local residual offsets in dim regions while bright central structures were well
aligned. A local diagnostic showed that the correct residual displacement produced a
larger correspondence improvement in gradient representations than in raw intensity.

The approved next step is to support ANTs neighborhood cross-correlation (CC) as an
explicit nonlinear metric, with a configurable scalar radius. The checked-in
workflow example will select CC. Radius 4 is a provisional default informed by ANTs
conventions, not a claim of dataset-specific optimality.

The current Phase 1 Python API remains a compatibility surface. Its omitted
nonlinear metric must continue to mean the legacy Mattes/32 policy. Workflow schema
version 1 will instead require an explicit nonlinear metric so an old YAML document
cannot silently change scientific behavior.

## Authoritative ANTs Constraint

The `antsRegistration` CC metric accepts one unsigned integer radius. ANTs constructs
the metric radius by filling every image dimension with that scalar. ANTSpyX exposes
the same value as scalar `syn_sampling`; it does not expose a per-axis CC radius.

Consequently, Cy-ExM will support one scalar voxel radius. It will not advertise
anisotropic radii. A radius of 4 evaluates a 9-by-9-by-9 voxel neighborhood. For
anisotropic voxels, this neighborhood is anisotropic in physical units. Resampling
estimation images to isotropic spacing or bypassing `antsRegistration` with a custom
ITK metric would be separate scientific changes and are outside this design.

Primary references:

- [ANTs metric parser](https://github.com/ANTsX/ANTs/blob/main/Examples/antsRegistration.cxx#L317-L328)
- [ANTs CC metric construction](https://github.com/ANTsX/ANTs/blob/main/Examples/itkantsRegistrationHelper.hxx#L931-L945)
- [ANTsPy registration API](https://antspy.readthedocs.io/en/stable/registration.html)

## Goals

1. Require an explicit nonlinear metric in workflow schema version 1.
2. Support exactly `CC` and `mattes` nonlinear metric names initially.
3. Let CC omit its radius and normalize the omission to built-in integer 4.
4. Forward the normalized metric and sampling parameter to every whole-image and
   serial-chunked nonlinear ANTs call.
5. Include the effective metric policy in scientific plans, fingerprints, artifact
   requests, cache decisions, and status reconstruction.
6. Preserve the Phase 1 API's omitted-argument Mattes/32 behavior.
7. Preserve every existing transform, geometry, mask, schedule, interpolation,
   precision, histogram, threading, and artifact-lifecycle invariant.
8. Document radius 4 as provisional and make a later optimized default a separate,
   evidence-backed scientific change.

## Non-goals

- Do not implement anisotropic CC radii.
- Do not resample registration inputs to isotropic spacing.
- Do not implement gradient, LoG, MIND, or multivariate metrics in this change.
- Do not optimize the radius on production or approved datasets in this change.
- Do not change affine TRSAA or its Mattes/32 policy.
- Do not change SyN schedules, gradient step, flow smoothing, total-field smoothing,
  masks, chunk geometry, overlap, or histogram postprocessing.
- Do not change round-level scheduling or add chunk-level parallelism.
- Do not introduce schema version 2 or dual schema maintenance.
- Do not touch HPC jobs, production datasets, or active registration outputs.

## Alternatives Considered

### Explicit metric with optional flat radius

The selected YAML shape places `metric` and optional `radius` beside nonlinear
`type` and `accuracy`. It is concise, strict, and gives the metric-specific parameter
its scientific name.

### Nested metric mapping

A nested mapping such as `metric: {type: CC, radius: 4}` would be extensible, but it
adds structure not needed for the two supported policies. It is rejected for the
current schema.

### Implicit CC with only a radius

Making CC implicit would shorten YAML but conceal a scientific choice, make old
documents silently change behavior, and complicate future metric support. It is
rejected.

### Generic `syn_sampling`

Exposing ANTSpyX's `syn_sampling` name would use the same field for a CC radius and a
Mattes bin count. That is technically direct but scientifically ambiguous. It is
rejected in favor of metric-specific YAML and normalized internal policy.

## Manifest Schema Version 1

The linear stage remains unchanged. The nonlinear stage requires `metric` and allows
`radius` only when the metric is CC:

```yaml
registration:
  crop: false
  linear:
    type: TRSAA
    accuracy: high
  nonlinear:
    type: SyNOnly
    accuracy: high
    metric: CC
    # radius: 4  # Optional; omitted means 4.
```

The exact validation contract is:

- `metric` is required and must be the built-in string `CC` or `mattes`.
- For `CC`, `radius` is optional. Omission normalizes to built-in integer 4.
- An explicit CC radius must be a built-in integer greater than or equal to 1.
- Boolean, float, string, null, zero, negative, and integer-subclass radius values
  are invalid.
- For `mattes`, `radius` is rejected because it has no Mattes meaning.
- Mattes retains its fixed 32-bin policy; no YAML bin selector is added.
- Unknown keys remain errors under the strict loader.

An omitted CC radius and explicit `radius: 4` produce equal normalized manifest
models. They must therefore produce byte-identical canonical scientific JSON and the
same fingerprint.

Existing YAML documents must add either `metric: CC` or `metric: mattes`. No automatic
migration occurs. Existing scientific plans lack the revised exact fields and fail
closed under the current mismatch behavior, requiring a new output root or an
operator-managed archive/removal of disposable prior outputs.

## Configuration Models

The shared `RegistrationStageSpec` remains appropriate for the linear stage. A new
frozen nonlinear-stage model represents nonlinear-only policy without allowing
irrelevant fields on the linear model:

```python
@dataclass(frozen=True, slots=True)
class NonlinearRegistrationStageSpec:
    type: str
    accuracy: str
    metric: Literal["CC", "mattes"]
    radius: int | None
```

Normalized CC models always contain an integer radius, including when YAML omitted
it. Mattes models contain `radius=None`. `RegistrationSpec.nonlinear` uses this new
type. The public configuration namespace exports it alongside the existing models.

Directly constructed manifests and workflow tasks are revalidated with the same
exact-type, metric, and radius rules before image inspection, filesystem mutation,
scientific imports, or executor work.

## Canonical Metric Policy

One pure resolver is authoritative for effective nonlinear metric parameters. It
performs no scientific import and maps:

| Requested policy | Effective `syn_metric` | Effective `syn_sampling` |
| --- | --- | ---: |
| `CC`, omitted radius | `CC` | 4 |
| `CC`, radius N | `CC` | N |
| `mattes` | `mattes` | 32 |

The configuration parser, Phase 1 constructors, nonlinear kernel, request-policy
builders, chunk provenance, and workflow status must not duplicate these defaults.
Execution parameters and cache evidence derive from the same resolved value.

The resolver validates metric/radius combinations before ANTSpyX, NumPy, or other
scientific runtimes are imported.

## Phase 1 API Compatibility

The standalone registration APIs gain keyword-only parameters:

- orchestration classes: `nonlinear_metric="mattes"` and
  `nonlinear_radius=None`;
- low-level nonlinear kernel: `metric="mattes"` and `radius=None`.

Omitting both preserves the existing Mattes/32 call and completion-request evidence.
Selecting `metric="CC"` with no radius resolves to radius 4. Passing a radius with
Mattes is an error. Existing positional call compatibility is unchanged.

`ImageRegistration` stores the validated normalized policy. Its
`ChunkedImageRegistration` subclass inherits the same policy. Whole-image and every
chunk-local `_estimate_nonlinear` call pass both values through the backend boundary.

## Workflow Data Flow

The effective policy flows through these existing layers without adding a new
runtime dependency:

```text
strict YAML
  -> normalized DatasetManifest
  -> canonical scientific plan and fingerprint
  -> immutable RoundTask
  -> ImageRegistration or ChunkedImageRegistration
  -> RegistrationBackend.nonlinear
  -> nonlinear.register_image
  -> ants.registration
```

`round_task.py` passes the normalized task fields as `nonlinear_metric` and
`nonlinear_radius`. The backend remains a thin adapter and does not own defaults or
artifact state.

## Scientific Plan and Status

The canonical scientific plan records effective metric-specific values rather than
whether YAML happened to omit a default. CC records:

```json
{
  "type": "SyNOnly",
  "accuracy": "high",
  "metric": "CC",
  "radius": 4
}
```

Mattes records the fixed bin policy explicitly:

```json
{
  "type": "SyNOnly",
  "accuracy": "high",
  "metric": "mattes",
  "bins": 32
}
```

Status applies the same metric-specific exact-key validation when reading a plan,
reconstructs the normalized nonlinear model, and uses it for every expected
completion request. Workflow API snapshot comparisons include the metric and
effective parameter.

## Cache Identity and Provenance

`nonlinear_parameters` becomes metric-aware and records the effective
`syn_metric`/`syn_sampling` pair used by ANTSpyX. The pair participates in:

- whole-image nonlinear-stage parameters;
- every chunk-local nonlinear request;
- chunk-manifest policy evidence;
- full reconstructed warp evidence;
- workflow status reconstruction.

Registered scalars and transformed channels continue to depend on the nonlinear warp
identity through their upstream evidence. Changing a CC radius invalidates all local
or whole-image nonlinear warps and their descendants while retaining compatible
affine evidence. Changing the default in a future release changes the normalized
scientific plan for YAML that omitted radius and therefore fails closed against old
plans and nonlinear cache records.

No data file alone becomes reusable. Existing completion-record, atomic-write,
stale-record, and disposable-warp ownership rules remain unchanged.

## Failure Handling

- Missing or invalid YAML metric policy fails during strict manifest loading.
- Invalid directly constructed models fail before TIFF inspection or output-root
  mutation.
- Invalid Phase 1 metric/radius calls fail before scientific runtime loading.
- A runtime call and its request evidence cannot disagree because both use the same
  pure resolver.
- Existing plans with the old exact nonlinear mapping fail with a precise plan-schema
  mismatch; Cy-ExM never silently interprets them as CC.
- Unsupported anisotropic radius syntax is rejected rather than flattened or
  reordered.
- ANTs runtime errors retain the existing registration-stage failure and artifact
  cleanup behavior.

## Test-Driven Implementation

### Manifest and model unit tests

1. Parse CC with omitted radius and assert normalized radius 4.
2. Parse explicit positive CC radii exactly.
3. Parse explicit Mattes with `radius=None` and preserve the 32-bin policy.
4. Reject missing/unknown metrics and every invalid radius type or range.
5. Reject radius with Mattes and reject unknown nonlinear keys.
6. Apply the same checks to directly constructed manifest models.

### Planning and status unit tests

1. Prove omitted radius and explicit radius 4 generate identical scientific JSON and
   fingerprints.
2. Prove radius 2 and radius 4 generate different fingerprints.
3. Assert effective metric-specific fields in the canonical scientific plan.
4. Reject old or malformed exact plan mappings.
5. Reconstruct RoundTasks with the effective metric policy.
6. Include metric policy in workflow API snapshot comparisons and immutable-task
   validation.

### Registration and provenance unit tests

1. Prove the nonlinear kernel sends CC/default as `syn_metric="CC"` and
   `syn_sampling=4`.
2. Prove an explicit CC radius is forwarded exactly.
3. Prove omitted Phase 1 parameters still send Mattes/32.
4. Prove invalid combinations fail before ANTs loading.
5. Prove whole-image and serial-chunked orchestration forward the normalized policy.
6. Assert effective parameters in whole-image, local-chunk, chunk-manifest, and full
   warp requests.
7. Prove a radius change invalidates nonlinear descendants but preserves affine
   reuse.
8. Prove workflow status and runtime make the same reuse decision.

### Concrete integration test

A small deterministic real-ANTs test will execute bounded CC SyN registration with a
configured scalar radius. It will verify successful warp materialization, finite
output, and exact fixed-reference geometry and artifact contracts. It will not claim
that radius 4 is optimal or that the synthetic registration predicts performance on
the spheroid dataset.

No portable test accesses lab storage, submits a scheduler job, or silently depends
on the HPC environment.

## Documentation and Guidance

Implementation updates:

- `README.md` configuration, scientific-policy, and migration sections;
- `examples/spheroid-dataset.yaml`, selecting CC and demonstrating omitted radius;
- `docs/brightness-dependence-investigation.md`, replacing ambiguous NCC deltas with
  absolute current/corrected values and labeling radius 4 provisional;
- the relevant root, configuration, registration, workflow, tests, documentation,
  and example `AGENTS.md` files, adding only scope-specific rules.

Documentation will state that ANTs exposes a scalar voxel radius, that anisotropic
spacing creates a physically anisotropic CC neighborhood, and that arbitrary
dependent channels are not scientifically valid similarity inputs merely because
they exist.

## Future Radius Characterization

A later, separately approved study may sweep a bounded set of scalar radii on an
approved cropped dataset. It must compare spatially resolved three-dimensional
residuals by signal stratum, bright-region preservation, seam behavior, positive
warp Jacobians/no folding, runtime, and reproducibility. Global correlation alone is
insufficient.

If that evidence supports a new default, changing the resolver's default is an
intentional scientific-policy change. Tests, documentation, plans, and cache identity
must change together. Explicit YAML radii remain stable across such a default change.

## Acceptance Criteria

The implementation is accepted only when:

1. workflow schema version 1 requires `metric` and supports exactly CC and Mattes;
2. omitted CC radius normalizes to 4 and explicit radius 4 has identical identity;
3. only positive built-in scalar integer CC radii are accepted;
4. Phase 1 omitted arguments remain Mattes/32 compatible;
5. actual ANTs calls and artifact evidence use one shared resolved policy;
6. scientific plans, fingerprints, RoundTasks, status, whole-image requests, and
   chunked requests include the effective policy;
7. changing radius invalidates nonlinear descendants without invalidating compatible
   affine evidence;
8. the real-ANTs CC integration regression preserves geometry and artifact contracts;
9. README, examples, migration guidance, investigation notes, and applicable
   `AGENTS.md` files are synchronized;
10. targeted tests and all relevant portable project gates pass; and
11. no HPC job or production dataset is touched.
