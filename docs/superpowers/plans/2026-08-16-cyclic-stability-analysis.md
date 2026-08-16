# Cyclic Nuclear-Envelope and Organelle Stability Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publication-specific, restart-safe analysis that quantifies
transform-derived deformation, post-registration Lamin/Golgi/LAMP1 morphology, and
repeated-labeling signal behavior across five 60x Cy-ExM rounds.

**Architecture:** Keep a standalone analysis Git repository in the external
resubmission workspace. Thin PEP 723 entry scripts call one focused Python package;
strict scientific configuration, target-blind registration QC, one marker-cell per
process, atomic task records, compact cell-level tables, and deterministic figure and
report rendering form the end-to-end boundary. Consume Cy-ExM's public workflow
status, artifact, transform, and geometry interfaces at the pinned registration
commit without changing the Cy-ExM runtime.

**Tech Stack:** Python 3.12, PEP 723 and uv, Cy-ExM pinned at
`c8b6a9f4a2f5f5fab3296fdc81190a7324da2a52`, ANTSpyX, NumPy, SciPy,
scikit-image, tifffile, nd2 0.11.3, PyYAML, pandas, Matplotlib, seaborn,
Jinja2, psutil, pytest, Ruff, Git, and optional SLURM arrays.

**Approved design:**
`docs/superpowers/specs/2026-08-16-cyclic-stability-analysis-design.md`

## Global Constraints

- The implementation root is exactly
  `/archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/cyclic-stability-analysis`.
- Source ND2 files, exported TIFFs, Cy-ExM manifests, Cy-ExM artifacts, and their
  completion records are read-only.
- The Cy-ExM repository receives no runtime, test, data, or generated-output changes
  from this analysis; this plan and its approved design specification are its only
  artifacts.
- Python is exactly the Cy-ExM-supported range `>=3.12,<3.13`.
- Every user-run entry script contains a complete PEP 723 metadata block.
- Public physical spacing is micrometers in `(x, y, z)` order; NumPy volumes are
  `(z, y, x)`. Warp arrays are `(z, y, x, component)` with displacement components
  in physical `(x, y, z)` order.
- The fixed biological reference is acquisition round 2, even where the Golgi
  workflow labels its fixed output as workflow round 1.
- Use nominal expansion factor `4.5`; all divided native-space values are labeled
  nominal estimates.
- Use theoretical diffraction references, not measured resolution: lateral
  `0.61 * wavelength / NA` and axial `2 * refractive_index * wavelength / NA**2`.
- The 640-nm target reference with NA `1.4` and refractive index `1.515` is about
  0.279 micrometers lateral and 0.989 micrometers axial in expanded-image space, or
  0.062 and 0.220 micrometers in nominal native space.
- Keep lateral and axial tolerances separate through an anisotropic normalized
  distance; do not replace them with one isotropic Euclidean tolerance.
- Continuous-image measurements are primary. Mask metrics use the entire frozen
  threshold/scale ensemble, and Jaccard remains secondary.
- Whole-cell volume and morphology are outside this analysis; DAPI defines only the
  nuclear and nucleus-relative coordinate systems.
- Cell is the biological replicate. Voxels, surface vertices, and objects are
  summarized within cells before group summaries.
- Rounds 2--5 are the matched-settings signal analysis. Round 1 is a separately
  labeled exposure-normalized sensitivity analysis because target exposure changed
  from 150 ms to 200 ms.
- No per-round intensity normalization may erase a genuine round-dependent signal
  change. Normalization used for a consensus surface or image-similarity feature is
  never substituted for the unregistered absolute signal table.
- LAMP1 production analysis remains blocked until five cross-round cell identities
  are established from image content and recorded; numeric ND2 suffixes are never
  treated as cell identities.
- Registration decisions are made from DAPI, tubulin, transforms, and completion
  evidence before target outcomes are computed. No universal correlation threshold
  is introduced.
- A target-blind `accept` or `exclude` decision with a reason is required for every
  non-reference cell-round before primary morphology aggregation.
- One marker-cell across all five rounds is the independent process-queue or SLURM
  array unit. Do not add nested multiprocessing.
- Default every ITK, OpenMP, and BLAS runtime to one native thread per process.
- Select production concurrency only after one full marker-cell pilot records peak
  RSS, elapsed time, process read bytes, and scratch use. Cap local workers at three
  even if memory arithmetic permits more, to bound concurrent archive I/O.
- Task outputs are restart-safe and atomic. Reuse requires exact config and input
  identities. Failed attempts remain diagnostic and never authorize reuse.
- Figures use an explicit TrueType font, editable vector text, a colorblind-safe
  palette, vector PDF/SVG, and high-resolution PNG previews. The centralized style
  sets `pdf.fonttype = 42`, `ps.fonttype = 42`, and `svg.fonttype = "none"`.
- Keep active Python environments and uv caches on `/project`, never `/archive`.
  Before every `uv` command export exactly:

  ```bash
  export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
  export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
  ```

- Preserve unrelated changes in the Cy-ExM worktree. In the external analysis Git
  repository, stage only files named by the current task.
- Do not claim that a nonsignificant round effect proves stability, that relabeling
  intensity isolates epitope chemistry, or that post-registration morphology proves
  an absence of pre-registration deformation.

## Scope check

The plan keeps one shared implementation rather than separate marker plans because
all marker modules consume the same audited correspondence, acquisition numbering,
artifact gate, target-blind QC decisions, task lifecycle, statistical unit, and
figure/report contract. Lamin, Golgi, and LAMP1 remain separate task boundaries and
can each be rejected or revised without changing the other marker algorithms.

## File and responsibility map

All paths in this section are relative to the implementation root unless an absolute
path is shown.

- Create `.gitignore`: ignore immutable run outputs, smoke crops, resource logs,
  Python caches, and editor files while retaining source and frozen configuration.
- Create `README.md`: scientific boundaries, exact commands, run lifecycle, QC gate,
  output inventory, and interpretation limits.
- Create `config/analysis.yaml`: frozen scientific parameters, paths, optics,
  threshold/scale ensembles, and marker policies.
- Create `config/correspondence.csv`: one exact row per audited Golgi and Lamin
  cell-round, with LAMP1 rows added only after image-based identity resolution.
- Create `config/qc-decisions.csv`: target-blind human decisions; begin with the
  schema header and no fabricated decisions.
- Create `config/execution-local.yaml`: sequential/process settings, scratch root,
  native threads, and a conservative pre-pilot worker cap.
- Create `config/execution-slurm.yaml`: optional one-cell-per-array-index execution
  settings with bounded array concurrency.
- Create `scripts/validate_inputs.py`: PEP 723 validation and provenance entry point.
- Create `scripts/run_registration_qc.py`: PEP 723 target-blind QC entry point.
- Create `scripts/run_analysis.py`: PEP 723 marker-cell execution entry point.
- Create `scripts/render_report.py`: PEP 723 aggregation, figure, and report entry
  point.
- Create `scripts/run_tests.py`: PEP 723 test entry point containing the complete
  development dependency set.
- Create `scripts/cyclic_stability/models.py`: immutable configuration, artifact,
  QC, task, and result value objects.
- Create `scripts/cyclic_stability/config.py`: strict YAML/CSV parsing and
  cross-field validation.
- Create `scripts/cyclic_stability/provenance.py`: ND2-to-TIFF equality audit and
  source identity capture.
- Create `scripts/cyclic_stability/cyexm_artifacts.py`: Cy-ExM status, layout,
  completion-record, geometry, and transform adapter.
- Create `scripts/cyclic_stability/units.py`: axis conversion, nominal expansion,
  and anisotropic theoretical-resolution calculations.
- Create `scripts/cyclic_stability/images.py`: read-only TIFF loading, physical
  smoothing, robust background, normalization, projections, and thumbnails.
- Create `scripts/cyclic_stability/registration_qc.py`: affine, warp, Jacobian,
  seam/interior, metric, and DAPI/tubulin overlay evidence.
- Create `scripts/cyclic_stability/morphology.py`: shared surface, distance,
  threshold-ensemble, and resolution-tolerant primitives.
- Create `scripts/cyclic_stability/signal.py`: unregistered target-channel signal
  summaries and round-1 exposure sensitivity.
- Create `scripts/cyclic_stability/nuclear.py`: consensus DAPI surface and Lamin
  normal-profile measurements.
- Create `scripts/cyclic_stability/golgi.py`: multiscale continuous similarity,
  threshold robustness, and nucleus-relative Golgi geometry.
- Create `scripts/cyclic_stability/lysosome.py`: scale-normalized LAMP1 detection,
  object matching, and loss/new summaries.
- Create `scripts/cyclic_stability/task.py`: one marker-cell scientific task.
- Create `scripts/cyclic_stability/runner.py`: sequential, bounded process, SLURM
  index, atomic attempt, restart, and resource-evidence orchestration.
- Create `scripts/cyclic_stability/aggregate.py`: QC-aware cell-level aggregation,
  reference and adjacent changes, descriptive slopes, and sensitivity tables.
- Create `scripts/cyclic_stability/figures.py`: centralized style and deterministic
  manuscript/QC figures.
- Create `scripts/cyclic_stability/report.py`: Markdown methods/results rendering
  from compact tables and run metadata.
- Create `scripts/cyclic_stability/templates/report.md.j2`: explicit manuscript and
  rebuttal language with assumption and exclusion sections.
- Create focused tests under `tests/` matching each module, plus `tests/data/` for
  small synthetic arrays and generated completion records.
- Create `tests/conftest.py`: shared temporary scientific/configuration fixtures.
- Create `tests/synthetic.py`: deterministic physical-space generators used by the
  focused marker, transform, executor, and report tests.
- Create `results/RUN_ID/` only at runtime. It contains `config/`, `validation/`,
  `qc/`, `tasks/`, `tables/`, `figures/`, `report.md`, `run.json`, and atomic
  completion records; the directory is ignored by the external source repository.

## Frozen initial scientific configuration

`config/analysis.yaml` starts with these exact values. Changing any value creates a
new run identity and requires regenerating every affected cell task.

```yaml
schema_version: 1
reference_acquisition_round: 2
voxel_spacing_xyz_um: [0.108333, 0.108333, 0.300000]
nominal_expansion_factor: 4.5
optics:
  numerical_aperture: 1.4
  immersion_refractive_index: 1.515
  wavelength_um:
    dapi: 0.405
    tubulin: 0.488
    target: 0.640
ensembles:
  gaussian_sigma_um: [0.279, 0.558, 1.116]
  threshold_sbr: [3.0, 4.0, 5.0]
  surface_tolerance_resolution_units: [0.5, 1.0, 1.5]
nuclear:
  consensus_smoothing_um: 0.558
  normal_half_window_um: 1.5
  normal_step_um: 0.05
  ridge_min_sbr: 3.0
golgi:
  perinuclear_outer_distance_um: 18.0
lysosome:
  log_sigma_um: [0.35, 0.70, 1.40]
  detection_sbr: [3.0, 4.0, 5.0]
  match_resolution_units: 1.0
paths:
  nd2_root: /archive/bioinformatics/Danuser_lab/Dean/Seweryn/resubmission/U2OS_repeat_60x
  resubmission_root: /archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission
  correspondence_markdown: /archive/bioinformatics/Danuser_lab/Dean/publication/2025-Galecki/Resubmission/nd2-tiff-file-correspondence.md
```

The ensemble values are analysis inputs, not optimized outcomes. The report shows
curves for every ensemble member and uses the median across the frozen ensemble as
the robust mask-derived summary.

## Reuse analysis

- Closest Cy-ExM workflow symbols: `load_dataset_manifest`,
  `inspect_workflow_status`, `WorkflowLayout`, and `RoundLayout`.
- Decision: use them to resolve dataset identity, authoritative round status, and
  fixed-grid channel locations. Do not parse workflow logs or invent a second status
  model.
- Closest Cy-ExM artifact symbols: `ArtifactStore.record_path`, `CompletionRecord`,
  and `FileIdentity`.
- Decision: use those symbols to verify every consumed artifact against its matching
  completion record. Analysis task records are publication-local because they
  authorize only derived tables, not Cy-ExM scientific reuse.
- Closest Cy-ExM transform and geometry symbols: `AntsRegistrationBackend`,
  `RegistrationPaths`, `ImageGeometry`, and `compute_chunk_grid`.
- Decision: read transforms through the backend and preserve the package's ZYX/XYZ
  adapters. Read the persisted chunk manifest for the actual production grid rather
  than recomputing it when classifying seams.
- Existing `tests/hpc/fixed_affine_chunk_analysis.py` demonstrates Jacobian and seam
  characterization but is test-only and not a supported import surface.
- Decision: implement publication-local, tested summaries while composing the public
  Cy-ExM readers. Do not import from `tests/`, recreate registration, or add a wrapper
  that renames a Cy-ExM lifecycle owner.
- Closest execution symbols are Cy-ExM's sequential, Local-Dask, and SLURM workflow
  executors.
- Decision: do not reuse them because this plan schedules small, read-only analysis
  tasks rather than registration rounds. Use one standard-library process queue and
  an optional caller-submitted SLURM array; neither may modify Cy-ExM artifacts.
- No Cy-ExM user-callable API or extension boundary changes. The external functions
  below are private to this publication and use concise typed docstrings rather than
  expanding the Cy-ExM public API.

---

### Task 1: Standalone PEP 723 project and strict scientific manifest

**Files:**

- Create: `.gitignore`
- Create: `README.md`
- Create: `config/analysis.yaml`
- Create: `config/correspondence.csv`
- Create: `config/qc-decisions.csv`
- Create: `config/execution-local.yaml`
- Create: `config/execution-slurm.yaml`
- Create: `scripts/run_tests.py`
- Create: `scripts/cyclic_stability/__init__.py`
- Create: `scripts/cyclic_stability/models.py`
- Create: `scripts/cyclic_stability/config.py`
- Create: `tests/conftest.py`
- Create: `tests/synthetic.py`
- Test: `tests/test_config.py`

**Interfaces:**

- Produces `AnalysisConfig`, `OpticsConfig`, `EnsembleConfig`, `NuclearConfig`,
  `GolgiConfig`, `LysosomeConfig`, `CellSpec`, `RoundSpec`, `ExecutionProfile`,
  `QcDecision`, `MetricRow`, `EnsembleRow`, and `MarkerResult` as frozen, slotted
  dataclasses. `MarkerResult` contains only one cell's summary rows, ensemble rows,
  diagnostics, and input identities.
- Produces
  `load_analysis_config(path: Path, correspondence_path: Path, qc_path: Path) -> AnalysisConfig`.
- Produces `load_execution_profile(path: Path) -> ExecutionProfile`.
- `RoundSpec.workflow_round` is explicit and never derived from acquisition round.
- Later tasks consume only validated value objects, never raw YAML or CSV mappings.
- Produces the `config_files` pytest fixture in `tests/conftest.py`; it writes the
  frozen YAML plus a complete five-cell-by-five-round synthetic Golgi CSV and empty
  QC-decision CSV into `tmp_path` and returns those three paths.

- [ ] **Step 1: Initialize the external repository and create the failing tests**

Run from the resubmission root:

```bash
mkdir -p cyclic-stability-analysis
cd cyclic-stability-analysis
git init -b kdean/cyclic-stability-analysis
mkdir -p config scripts/cyclic_stability tests
```

Create `scripts/run_tests.py` with the full PEP 723 environment and a pytest bridge:

```python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@c8b6a9f4a2f5f5fab3296fdc81190a7324da2a52",
#   "nd2==0.11.3",
#   "numpy>=1.26,<3",
#   "scipy>=1.12,<2",
#   "scikit-image>=0.22,<1",
#   "tifffile>=2025.1.10",
#   "PyYAML>=6,<7",
#   "pandas>=2.2,<3",
#   "matplotlib>=3.8,<4",
#   "seaborn>=0.13,<1",
#   "Jinja2>=3.1,<4",
#   "psutil>=5.9,<8",
#   "pytest>=8,<9",
# ]
# ///
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
raise SystemExit(pytest.main([str(ROOT / "tests"), *sys.argv[1:]]))
```

Create `tests/conftest.py`, `tests/synthetic.py`, and `tests/test_config.py` with
strict round-numbering and physics checks. The `config_files` fixture uses the same
column order and Golgi round permutation as the production correspondence CSV.

```python
from pathlib import Path

import pytest

from cyclic_stability.config import load_analysis_config


def test_config_preserves_acquisition_and_workflow_rounds(config_files: tuple[Path, ...]) -> None:
    analysis, correspondence, decisions = config_files
    config = load_analysis_config(analysis, correspondence, decisions)
    golgi = config.cell("golgi", "Cell1")
    assert [item.acquisition_round for item in golgi.rounds] == [1, 2, 3, 4, 5]
    assert [item.workflow_round for item in golgi.rounds] == [2, 1, 3, 4, 5]
    assert config.reference_acquisition_round == 2
    assert config.nominal_expansion_factor == 4.5


def test_config_rejects_numeric_lamp_identity(tmp_path: Path, config_files: tuple[Path, ...]) -> None:
    analysis, correspondence, decisions = config_files
    text = correspondence.read_text(encoding="utf-8")
    text += (
        "lamp1,Cell1,1,1,DAPI_tubulin_lamp_003.nd2,LAMP1/Cell1,0,"
        "647_lamp1,150,50,200,50,200,75,lamp.yaml,exec.yaml,lamp-cell1,"
        "numeric-suffix\n"
    )
    invalid = tmp_path / "correspondence.csv"
    invalid.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="LAMP1 identity must be image-verified"):
        load_analysis_config(analysis, invalid, decisions)
```

- [ ] **Step 2: Run the tests and verify the missing package fails**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_config.py -q
```

Expected: collection fails because `cyclic_stability.config` does not exist.

- [ ] **Step 3: Implement immutable models and strict parsing**

Define the central round and analysis values in `models.py`:

```python
@dataclass(frozen=True, slots=True)
class RoundSpec:
    cohort: Literal["golgi", "lamin", "lamp1"]
    cell_id: str
    acquisition_round: int
    workflow_round: int
    nd2_path: Path
    tiff_directory: Path
    z_start_offset: int
    target_exposure_ms: float
    target_laser_percent: float
    tubulin_exposure_ms: float
    dapi_exposure_ms: float
    dataset_manifest: Path
    execution_profile: Path
    workflow_dataset_id: str
    identity_method: Literal["exact-pixel-audit", "image-verified"]


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    reference_acquisition_round: int
    voxel_spacing_xyz_um: tuple[float, float, float]
    nominal_expansion_factor: float
    optics: OpticsConfig
    ensembles: EnsembleConfig
    nuclear: NuclearConfig
    golgi: GolgiConfig
    lysosome: LysosomeConfig
    cells: tuple[CellSpec, ...]
    qc_decisions: tuple[QcDecision, ...]
    config_digest: str

    def cell(self, cohort: str, cell_id: str) -> CellSpec:
        matches = [c for c in self.cells if c.cohort == cohort and c.cell_id == cell_id]
        if len(matches) != 1:
            raise KeyError(f"expected one cell for {cohort}/{cell_id}; got {len(matches)}")
        return matches[0]
```

Reject unknown keys, booleans in numeric fields, non-finite or non-positive physical
values, duplicate cell-rounds, missing rounds, acquisition rounds outside 1--5,
target names outside the three cohorts, inconsistent settings within a cohort, and
any Golgi/Lamin cohort that does not contain exactly five cells by five rounds.
Require LAMP1 rows to use `identity_method=image-verified` and five complete rounds
per included cell.

Transcribe the 50 audited Golgi/Lamin rows from
`nd2-tiff-file-correspondence.md` into `config/correspondence.csv`. Include these
columns exactly:

```text
cohort,cell_id,acquisition_round,workflow_round,nd2_relative,tiff_directory_relative,z_start_offset,target_channel,target_exposure_ms,target_laser_percent,tubulin_exposure_ms,tubulin_laser_percent,dapi_exposure_ms,dapi_laser_percent,dataset_manifest,execution_profile,workflow_dataset_id,identity_method
```

For Golgi, encode acquisition-to-workflow rounds `1:2, 2:1, 3:3, 4:4, 5:5`.
For Lamin, encode `1:1, 2:2, 3:3, 4:4, 5:5`. Require the source ND2 relative
paths and Z offsets to match the audited Markdown exactly. Start
`config/qc-decisions.csv` with only this header:

```text
cohort,cell_id,acquisition_round,decision,reasons,reviewer,reviewed_at
```

Create `config/execution-local.yaml` with `mode: sequential`, `max_workers: 2`,
`native_threads: 1`, and scratch root
`/project/bioinformatics/Danuser_lab/Dean/dean/cyclic-stability-analysis-scratch`.
Create the optional SLURM profile with one process per array element, array
concurrency 2, 1 CPU, 64 GiB, 12 hours, and the same native-thread value. These are
pre-pilot conservative settings, not measured resource requirements.

- [ ] **Step 4: Run focused tests and inspect the normalized inventory**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_config.py -q
```

Expected: PASS; the normalized inventory contains 25 Golgi and 25 Lamin rounds, no
eligible LAMP1 cells, and the exact Golgi acquisition/workflow permutation.

- [ ] **Step 5: Document and commit the foundation**

Document the data boundary, exact uv commands, target-blind QC pause, run identity,
and output layout in `README.md`. Ignore `results/`, `smoke-data/`, `.partial/`,
`__pycache__/`, `.pytest_cache/`, and editor files in `.gitignore`.

```bash
git add .gitignore README.md config scripts/run_tests.py scripts/cyclic_stability tests/conftest.py tests/synthetic.py tests/test_config.py
git commit -m "feat: establish cyclic stability analysis manifest"
```

---

### Task 2: Source correspondence and Cy-ExM artifact validation

**Files:**

- Create: `scripts/validate_inputs.py`
- Create: `scripts/cyclic_stability/provenance.py`
- Create: `scripts/cyclic_stability/cyexm_artifacts.py`
- Modify: `tests/synthetic.py`
- Test: `tests/test_provenance.py`
- Test: `tests/test_cyexm_artifacts.py`

**Interfaces:**

- Consumes `AnalysisConfig`, `CellSpec`, and `RoundSpec` from Task 1.
- Produces `PlaneReader` as a protocol with `nd2_plane(...)` and `tiff_plane(...)`
  methods and
  `compare_nd2_export(round_spec: RoundSpec, reader: PlaneReader | None = None) -> tuple[ChannelMatch, ...]`.
- Defines
  `StatusInspector = Callable[[str | Path, str], WorkflowStatus]`; the production
  default is Cy-ExM's `inspect_workflow_status`, and the test fixture returns one
  complete synthetic status.
- Produces
  `resolve_round_artifacts(round_spec: RoundSpec, status_inspector: StatusInspector = inspect_workflow_status) -> RoundArtifacts`.
- Produces
  `validate_cell_inputs(cell: CellSpec) -> CellValidationReport`.
- `RoundArtifacts` contains registered DAPI/tubulin/target paths, affine, full warp,
  pre/linear/nonlinear metric paths, chunk manifest, and matching completion records.
- Reference-round `RoundArtifacts.affine` and `.warp` are `None`; its channels still
  require valid completion records.
- Extend `tests/synthetic.py` with
  `write_completion_fixture(path: Path, payload: bytes) -> Path`; it uses
  `ArtifactStore.commit` and an exact synthetic `StageRequest`. Use it to construct
  the `completed_artifact` fixture. Build `fake_round`, `fake_plane_reader`, and
  `golgi_round_two` from the validated Task 1 dataclasses with temporary TIFF and
  workflow trees; the fake reader supplies exact in-memory ND2 planes without
  pretending a TIFF file is an ND2 acquisition.

- [ ] **Step 1: Write failing correspondence and artifact tests**

```python
def test_exact_export_match_checks_three_channels_and_three_planes(
    fake_round: RoundSpec, fake_plane_reader: PlaneReader
) -> None:
    matches = compare_nd2_export(fake_round, reader=fake_plane_reader)
    assert [(m.channel, m.plane_indices) for m in matches] == [
        ("target", (0, 2, 4)),
        ("tubulin", (0, 2, 4)),
        ("dapi", (0, 2, 4)),
    ]
    assert all(m.exact for m in matches)


def test_completion_identity_mismatch_is_rejected(completed_artifact: Path) -> None:
    completed_artifact.write_bytes(completed_artifact.read_bytes() + b"changed")
    with pytest.raises(InputValidationError, match="completion identity mismatch"):
        validate_completed_artifact(completed_artifact)


def test_golgi_reference_resolves_acquisition_round_two_from_workflow_round_one(
    golgi_round_two: RoundSpec, complete_status_inspector: StatusInspector,
) -> None:
    artifacts = resolve_round_artifacts(golgi_round_two, complete_status_inspector)
    assert artifacts.acquisition_round == 2
    assert artifacts.workflow_round == 1
    assert artifacts.registered_target.name == "round_2_647_golgi.tif"
    assert artifacts.affine is None
```

- [ ] **Step 2: Run the focused tests and verify missing adapters fail**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_provenance.py tests/test_cyexm_artifacts.py -q
```

Expected: FAIL because the provenance and artifact adapters are absent.

- [ ] **Step 3: Implement exact source and completion validation**

Use `nd2.ND2File` to read only the audited target/GFP/DAPI planes after applying
`z_start_offset`. Compare TIFF and ND2 values with `numpy.array_equal`; never use a
similarity tolerance for source correspondence. Return the exact plane indices and
channel names in `ChannelMatch`.

Compose Cy-ExM lifecycle owners directly:

```python
def validate_completed_artifact(path: Path) -> CompletionRecord:
    store = ArtifactStore()
    record_path = store.record_path(path)
    if not path.is_file() or not record_path.is_file():
        raise InputValidationError(f"artifact or completion record missing: {path}")
    record = CompletionRecord.read(record_path)
    if FileIdentity.from_path(path) != record.artifact:
        raise InputValidationError(f"completion identity mismatch: {path}")
    return record


def require_complete_workflow(round_spec: RoundSpec) -> WorkflowStatus:
    profile = load_execution_profile(round_spec.execution_profile)
    status = inspect_workflow_status(profile.paths.output_root, round_spec.workflow_dataset_id)
    if not status.complete:
        raise InputValidationError(
            f"Cy-ExM workflow is not complete: {round_spec.workflow_dataset_id}"
        )
    return status
```

Resolve paths with `WorkflowLayout.for_dataset(...).round(workflow_round)`,
`RegistrationPaths.for_round(round_layout.registration, workflow_round)`, and the
exact acquisition-round channel basenames. Validate every consumed channel,
transform, warp, metrics file, and chunk manifest before returning it.

- [ ] **Step 4: Add the PEP 723 validation CLI and run tests**

`scripts/validate_inputs.py` uses the same runtime dependency block as
`run_tests.py` without pytest, accepts `--analysis-config`, `--correspondence`,
`--qc-decisions`, `--cohort`, `--cell`, `--run-dir`, and `--skip-nd2-pixel-check`,
and writes `validation/inputs.json` atomically. The skip flag remains visibly false
in production run metadata and is allowed only for synthetic tests.

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_provenance.py tests/test_cyexm_artifacts.py -q
```

Expected: PASS, including exact mismatch and acquisition/workflow round tests.

- [ ] **Step 5: Run a read-only real-data validation smoke and commit**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/validate_inputs.py \
  --analysis-config config/analysis.yaml \
  --correspondence config/correspondence.csv \
  --qc-decisions config/qc-decisions.csv \
  --cohort golgi --cell Cell1 \
  --run-dir results/validation-smoke
```

Expected: all 15 source channel checks and every required Cy-ExM completion identity
for Golgi Cell1 pass without modifying source or registration trees.

```bash
git add scripts/validate_inputs.py scripts/cyclic_stability/provenance.py scripts/cyclic_stability/cyexm_artifacts.py tests/synthetic.py tests/test_provenance.py tests/test_cyexm_artifacts.py
git commit -m "feat: validate cyclic imaging provenance"
```

---

### Task 3: Physical units and target-blind registration/deformation QC

**Files:**

- Create: `scripts/run_registration_qc.py`
- Create: `scripts/cyclic_stability/units.py`
- Create: `scripts/cyclic_stability/registration_qc.py`
- Modify: `tests/synthetic.py`
- Test: `tests/test_units.py`
- Test: `tests/test_registration_qc.py`

**Interfaces:**

- Consumes validated `RoundArtifacts` from Task 2.
- Produces
  `theoretical_resolution(wavelength_um: float, na: float, refractive_index: float, expansion: float) -> ResolutionReference`.
- Produces
  `decompose_affine(matrix_xyz: NDArray[np.float64]) -> AffineSummary`.
- Produces
  `summarize_warp(warp_zyx_xyz: NDArray[np.float64], spacing_xyz_um: tuple[float, float, float], resolution: ResolutionReference) -> WarpSummary`.
- Produces
  `analyze_registration_qc(cell: CellSpec, artifacts: tuple[RoundArtifacts, ...], output_dir: Path) -> RegistrationQcResult`.
- `RegistrationQcResult` contains cell-level CSV rows, per-round automated validity,
  DAPI/tubulin overlay paths, seam/interior summaries, and a blank decision request
  for every moving round.
- Extend `tests/synthetic.py` with
  `synthetic_folded_warp(shape: tuple[int, int, int]) -> NDArray[np.float64]`;
  it returns a ZYX/XYZ displacement whose X derivative is less than `-1` in the
  central slab. Construct `TARGET_RESOLUTION` in the test module through
  `theoretical_resolution(0.640, 1.4, 1.515, 4.5)`.

- [ ] **Step 1: Write failing physical-resolution and deformation tests**

```python
def test_target_resolution_is_labeled_in_expanded_and_nominal_space() -> None:
    value = theoretical_resolution(0.640, 1.4, 1.515, 4.5)
    assert value.lateral_expanded_um == pytest.approx(0.278857, rel=1e-5)
    assert value.axial_expanded_um == pytest.approx(0.989388, rel=1e-5)
    assert value.lateral_nominal_native_um == pytest.approx(0.061968, rel=1e-5)
    assert value.axial_nominal_native_um == pytest.approx(0.219864, rel=1e-5)
    assert value.kind == "theoretical"


def test_constant_physical_warp_has_expected_resolution_fraction() -> None:
    warp = np.zeros((5, 6, 7, 3), dtype=np.float64)
    warp[..., 0] = 0.278857
    summary = summarize_warp(warp, (0.108333, 0.108333, 0.3), TARGET_RESOLUTION)
    assert summary.displacement_p95_um == pytest.approx(0.278857)
    assert summary.fraction_over_one_resolution == pytest.approx(0.0)
    assert summary.jacobian_nonpositive_count == 0


def test_folded_warp_is_not_silently_summarized_as_valid() -> None:
    warp = synthetic_folded_warp(shape=(9, 9, 9))
    summary = summarize_warp(warp, (0.1, 0.1, 0.3), TARGET_RESOLUTION)
    assert summary.jacobian_nonpositive_count > 0
    assert "nonpositive-jacobian" in summary.flags
```

- [ ] **Step 2: Run tests and verify the scientific functions are missing**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_units.py tests/test_registration_qc.py -q
```

Expected: FAIL because the unit and QC modules do not exist.

- [ ] **Step 3: Implement affine, warp, Jacobian, and seam summaries**

Read transforms through
`AntsRegistrationBackend(NativeThreadConfig(itk_threads=1, blas_threads=1))`.
Convert affine
parameters to a physical 4-by-4 XYZ matrix including center of rotation. Use polar
and RQ decomposition to report determinant, singular values, XYZ scale, anisotropy,
rotation, translation, and XY/XZ/YZ shear angles. Classify translation and rotation
as repositioning fields in output rows; do not include them in a morphology-stability
endpoint.

Build the deformation gradient explicitly from physical derivatives:

```python
def deformation_gradient(
    warp_zyx_xyz: NDArray[np.float64],
    spacing_xyz_um: tuple[float, float, float],
) -> NDArray[np.float64]:
    sx, sy, sz = spacing_xyz_um
    gradients = [np.gradient(warp_zyx_xyz[..., c], sz, sy, sx) for c in range(3)]
    field = np.zeros((*warp_zyx_xyz.shape[:3], 3, 3), dtype=np.float64)
    field[..., 0, :] = np.stack((gradients[0][2], gradients[0][1], gradients[0][0]), axis=-1)
    field[..., 1, :] = np.stack((gradients[1][2], gradients[1][1], gradients[1][0]), axis=-1)
    field[..., 2, :] = np.stack((gradients[2][2], gradients[2][1], gradients[2][0]), axis=-1)
    field += np.eye(3)
    return field
```

Define normalized resolution distance as
`sqrt((dx/lateral)**2 + (dy/lateral)**2 + (dz/axial)**2)`. Treat values equal to one
as within one reference unit; count only `> 1.0`. Parse actual core boundaries from
the persisted chunk manifest and classify voxels within 16 expanded-image pixels of
an internal core boundary as seam-adjacent. Report seam and interior displacement and
Jacobian summaries separately. Also report Frobenius-norm displacement-gradient
percentiles so local change is not reduced to displacement magnitude and Jacobian
alone.

- [ ] **Step 4: Implement blinded overlays and QC decisions**

Generate DAPI and tubulin XY/XZ/YZ overlays for pre-, affine-, and nonlinear stages
without loading target images. Read the three persisted Cy-ExM metric files through
`cyexm.metrics.read_metrics`. Automated exclusion is limited to invalid completion
evidence, non-finite geometry/data, shape mismatch, unreadable transforms, or a
non-positive Jacobian. Correlation and Mattes values are evidence columns, not
universal accept/reject thresholds.

Write `qc/registration-qc.csv`, `qc/deformation.csv`, and
`qc/decision-requests.csv`. The latter contains one row per non-reference cell-round
with `decision` empty. `run_registration_qc.py` accepts `--run-dir` and never imports
the target-specific modules.

- [ ] **Step 5: Run focused tests and the target-blind current-data QC**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_units.py tests/test_registration_qc.py -q
uv run --script scripts/run_registration_qc.py \
  --analysis-config config/analysis.yaml \
  --correspondence config/correspondence.csv \
  --run-dir results/qc-only
```

Expected: unit tests pass; QC produces 40 moving-round decision requests for the ten
current cells, explicitly surfaces the poor Lamin round-1 candidates, and reads no
Golgi or Lamin target channel.

- [ ] **Step 6: Pause for target-blind human review and commit**

Copy reviewed decisions into `config/qc-decisions.csv` only after the reviewer has
examined the DAPI/tubulin overlays and transform evidence. Each of the 40 rows must
contain `accept` or `exclude`, at least one reason token, reviewer, and ISO-8601
timestamp. Do not proceed to Tasks 4--9 production execution until this file passes
validation. Algorithm implementation and synthetic tests may continue while review
is pending.

```bash
git add config/qc-decisions.csv scripts/run_registration_qc.py scripts/cyclic_stability/units.py scripts/cyclic_stability/registration_qc.py tests/synthetic.py tests/test_units.py tests/test_registration_qc.py
git commit -m "feat: quantify target-blind registration deformation"
```

---

### Task 4: Shared image, morphology, and repeated-labeling signal primitives

**Files:**

- Create: `scripts/cyclic_stability/images.py`
- Create: `scripts/cyclic_stability/morphology.py`
- Create: `scripts/cyclic_stability/signal.py`
- Modify: `tests/synthetic.py`
- Test: `tests/test_images.py`
- Test: `tests/test_morphology.py`
- Test: `tests/test_signal.py`

**Interfaces:**

- Consumes physical spacing and resolution values from Task 3.
- Produces `gaussian_sigma_zyx(sigma_um: float, spacing_xyz_um: tuple[float, float, float]) -> tuple[float, float, float]`.
- Produces `robust_background(image: NDArray[np.floating]) -> BackgroundSummary`.
- Produces `threshold_ensemble(image, roi, sigma_values_um, threshold_sbr, spacing_xyz_um) -> tuple[MaskMember, ...]`.
- Produces `resolution_surface_metrics(fixed_mask, moving_mask, spacing_xyz_um, resolution, tolerance_units) -> SurfaceMetrics`.
- Produces `measure_signal(round_spec, target_zyx, structure_roi, ensemble) -> tuple[SignalRow, ...]`.
- Extend `tests/synthetic.py` with
  `single_voxel(shape_zyx, index_zyx)`, `synthetic_target()`, and
  `synthetic_roi()` returning deterministic NumPy arrays. In the test module define
  `SPACING = (0.1, 0.1, 0.3)`, construct `RESOLUTION` through Task 3, use the frozen
  nine-member `ENSEMBLE`, and construct `round_specs` from Task 1 dataclasses.

- [ ] **Step 1: Write failing anisotropic morphology and signal tests**

```python
def test_physical_gaussian_sigma_uses_zyx_voxel_order() -> None:
    assert gaussian_sigma_zyx(0.6, (0.1, 0.2, 0.3)) == pytest.approx((2.0, 3.0, 6.0))


def test_resolution_surface_dice_treats_lateral_and_axial_offsets_differently() -> None:
    fixed = single_voxel((9, 9, 9), (4, 4, 4))
    lateral = single_voxel((9, 9, 9), (4, 4, 5))
    axial = single_voxel((9, 9, 9), (5, 4, 4))
    lateral_result = resolution_surface_metrics(fixed, lateral, SPACING, RESOLUTION, 1.0)
    axial_result = resolution_surface_metrics(fixed, axial, SPACING, RESOLUTION, 1.0)
    assert lateral_result.normalized_distance_p95 != axial_result.normalized_distance_p95


def test_round_one_signal_is_kept_out_of_matched_setting_rows(round_specs: tuple[RoundSpec, ...]) -> None:
    rows = measure_signal(round_specs[0], synthetic_target(), synthetic_roi(), ENSEMBLE)
    assert all(row.comparison_set == "round-1-exposure-sensitivity" for row in rows)
    assert all(row.exposure_normalized for row in rows)
```

- [ ] **Step 2: Run tests and verify the primitives are absent**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_images.py tests/test_morphology.py tests/test_signal.py -q
```

Expected: FAIL because the shared modules are missing.

- [ ] **Step 3: Implement read-only image and threshold-ensemble operations**

Load TIFFs with `tifffile.memmap` when layout permits and otherwise with
`tifffile.imread`; require one finite 3D ZYX volume. Convert physical Gaussian sigma
to ZYX voxel sigma. Define background from positive voxels at or below their 30th
percentile and scale as `1.4826 * MAD`; reject an empty background or zero robust
scale rather than substituting a favorable value.

For every configured Gaussian sigma and SBR threshold, smooth only for mask
construction and define `mask = smoothed > background.median + sbr * background.scale`.
Use the original intensities for signal integration. Return every member plus the
member's exact sigma and threshold. The robust mask summary is the median of member
metrics; curves retain all members.

Surface metrics use physical coordinates from `skimage.measure.marching_cubes` and
`scipy.spatial.cKDTree`. Normalize XYZ deltas by `(lateral, lateral, axial)` before
the nearest-neighbor query. Report directed and symmetric boundary distances,
resolution-tolerant surface Dice for every configured tolerance, and Jaccard.

- [ ] **Step 4: Implement unregistered signal summaries**

Use each round's exported, unregistered target TIFF. Derive Lamin structure ROIs
from that round's raw DAPI nuclear surface and Golgi/LAMP1 spatial ROIs from that
round's raw DAPI nucleus-relative coordinates, so integrated signal is not confined
to a target-selected mask. Within the structure ROI and for every ensemble mask,
compute:

```python
corrected = np.clip(target_zyx.astype(np.float64) - background.median, 0.0, None)
integrated_per_ms = float(corrected[mask].sum() / round_spec.target_exposure_ms)
sbr = float((np.median(target_zyx[mask]) - background.median) / background.scale)
positive_volume_um3 = float(mask.sum() * np.prod(spacing_xyz_um))
```

Also store raw and corrected intensity percentiles 10, 25, 50, 75, 90, 95, and 99.
Mark acquisition round 1 as `round-1-exposure-sensitivity`; mark rounds 2--5 as
`matched-settings`. DAPI ratios are secondary rows and never replace absolute target
measurements.

- [ ] **Step 5: Run focused tests and commit**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_images.py tests/test_morphology.py tests/test_signal.py -q
git add scripts/cyclic_stability/images.py scripts/cyclic_stability/morphology.py scripts/cyclic_stability/signal.py tests/synthetic.py tests/test_images.py tests/test_morphology.py tests/test_signal.py
git commit -m "feat: add morphology and signal primitives"
```

---

### Task 5: Consensus nuclear surface and Lamin stability

**Files:**

- Create: `scripts/cyclic_stability/nuclear.py`
- Modify: `tests/synthetic.py`
- Test: `tests/test_nuclear.py`

**Interfaces:**

- Consumes registered DAPI/Lamin volumes, Task 4 morphology primitives, acquisition
  round 2 as reference, and frozen nuclear parameters.
- Produces
  `consensus_nuclear_surface(dapi_rounds: Mapping[int, NDArray], spacing_xyz_um, config) -> NuclearSurface`.
- Produces
  `lamin_normal_profiles(surface: NuclearSurface, lamin_rounds: Mapping[int, NDArray], config, background_by_round) -> tuple[LaminProfileRow, ...]`.
- Produces
  `analyze_lamin_cell(cell: CellSpec, artifacts, config, output_dir) -> MarkerResult`.
- `MarkerResult` contains cell-summary rows, ensemble rows, diagnostics, and no
  group-level inference.
- Extend `tests/synthetic.py` with
  `synthetic_nucleus(radius_um: float, shell_width_um: float) -> tuple[NDArray, NDArray]`,
  `synthetic_shell(radius_um: float, shell_width_um: float) -> NDArray`, and
  `synthetic_backgrounds(rounds: tuple[int, ...]) -> dict[int, BackgroundSummary]`.
  Use `SPACING = (0.1, 0.1, 0.3)` and the exact frozen `NUCLEAR_CONFIG` in the test
  module.

- [ ] **Step 1: Write failing shell-displacement, width, and missing-support tests**

```python
def test_lamin_profiles_recover_known_outward_displacement() -> None:
    dapi, reference_lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    moved_lamin = synthetic_shell(radius_um=5.6, shell_width_um=0.4)
    surface = consensus_nuclear_surface({2: dapi, 3: dapi}, SPACING, NUCLEAR_CONFIG)
    rows = lamin_normal_profiles(
        surface,
        {2: reference_lamin, 3: moved_lamin},
        NUCLEAR_CONFIG,
        synthetic_backgrounds((2, 3)),
    )
    round_three = [row for row in rows if row.acquisition_round == 3]
    assert np.median([row.displacement_um for row in round_three]) == pytest.approx(0.6, abs=0.12)


def test_missing_lamin_peak_is_missing_not_zero_displacement() -> None:
    dapi, reference_lamin = synthetic_nucleus(radius_um=5.0, shell_width_um=0.4)
    surface = consensus_nuclear_surface({2: dapi, 4: dapi}, SPACING, NUCLEAR_CONFIG)
    rows = lamin_normal_profiles(
        surface,
        {2: reference_lamin, 4: np.zeros_like(reference_lamin)},
        NUCLEAR_CONFIG,
        synthetic_backgrounds((2, 4)),
    )
    round_four = [row for row in rows if row.acquisition_round == 4]
    assert all(row.supported is False for row in round_four)
    assert all(math.isnan(row.displacement_um) for row in round_four)
```

- [ ] **Step 2: Run the test and verify the Lamin module is absent**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_nuclear.py -q
```

Expected: FAIL because `cyclic_stability.nuclear` is missing.

- [ ] **Step 3: Implement the common nuclear surface**

For each QC-accepted registered DAPI round, subtract robust background and divide by
the 99th percentile of positive corrected intensity. Take the voxelwise median,
smooth at `0.558` micrometers, threshold with Otsu, retain the largest 26-connected
component, fill enclosed holes, and close gaps with a ball of radius one theoretical
DAPI lateral resolution. Reject nuclei touching more than 10 percent of any volume
face.

Create the surface with marching cubes. Construct outward normals from the gradient
of the signed physical distance field and orient them away from the mask centroid.
Store vertices in physical XYZ micrometers and retain the mask for secondary volume,
area-to-volume, sphericity, and curvature summaries.

- [ ] **Step 4: Implement Lamin normal profiles and ensemble masks**

Sample each registered Lamin round from `-1.5` to `+1.5` micrometers along each
surface normal in `0.05`-micrometer steps with trilinear interpolation. A supported
ridge must exceed that round's robust background by SBR 3.0 and have its peak inside
the search interval rather than on an endpoint. Relative to acquisition round 2,
report signed peak displacement, absolute displacement, FWHM, integrated ridge
intensity, surface coverage, and continuity. Unsupported profiles contain NaN
geometry and contribute to coverage loss.

Calculate FWHM by linear interpolation at half the background-subtracted peak.
Coverage is supported vertices divided by sampled vertices. Continuity is the
fraction of sampled vertices in the largest edge-connected supported surface patch;
the primary continuity endpoint is its change from acquisition round 2. Calculate
secondary mean curvature as the divergence of the normalized signed-distance
gradient sampled at surface vertices.

Use Task 4's full threshold/scale ensemble for secondary Lamin masks. Report mask
volume, surface area-to-volume, sphericity, curvature quantiles,
resolution-tolerant surface Dice, boundary distances, and Jaccard. Produce XY/XZ/YZ
surface/profile diagnostics without choosing an ensemble member by outcome.

- [ ] **Step 5: Run tests and commit**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_nuclear.py tests/test_morphology.py -q
git add scripts/cyclic_stability/nuclear.py tests/synthetic.py tests/test_nuclear.py
git commit -m "feat: measure lamin envelope stability"
```

---

### Task 6: Golgi continuous similarity and morphology robustness

**Files:**

- Create: `scripts/cyclic_stability/golgi.py`
- Modify: `tests/synthetic.py`
- Test: `tests/test_golgi.py`

**Interfaces:**

- Consumes the consensus DAPI surface from Task 5, registered Golgi volumes, and the
  Task 4 threshold/scale ensemble.
- Produces
  `multiscale_gradient_similarity(reference, moving, roi, spacing_xyz_um, sigma_values_um) -> tuple[GradientSimilarityRow, ...]`.
- Produces
  `golgi_nucleus_distances(mask, nuclear_surface, spacing_xyz_um) -> NDArray[np.float64]`.
- Produces
  `analyze_golgi_cell(cell: CellSpec, artifacts, config, output_dir) -> MarkerResult`.
- Produces the array-level kernel
  `analyze_golgi_arrays(golgi_rounds: Mapping[int, NDArray], dapi_rounds: Mapping[int, NDArray], config: AnalysisConfig) -> MarkerResult`;
  the cell entry point owns I/O and delegates numerical work to this kernel.
- Extend `tests/synthetic.py` with
  `synthetic_golgi(components: int = 1)`, `synthetic_perinuclear_roi()`, and
  `synthetic_dapi_rounds()` returning deterministic physical-space arrays. In the
  test module, load `ANALYSIS_CONFIG` from the frozen Task 1 fixture and derive
  `SPACING`, `SIGMAS`, and `THRESHOLDS` from it.

- [ ] **Step 1: Write failing intensity-invariance and fragmentation tests**

```python
def test_gradient_similarity_is_robust_to_global_intensity_scaling() -> None:
    reference = synthetic_golgi()
    rows = multiscale_gradient_similarity(
        reference, reference * 2.5 + 100.0, synthetic_perinuclear_roi(), SPACING, SIGMAS
    )
    assert min(row.correlation for row in rows) > 0.99


def test_fragmentation_change_is_reported_for_every_threshold_member() -> None:
    result = analyze_golgi_arrays(
        {2: synthetic_golgi(components=1), 3: synthetic_golgi(components=4)},
        synthetic_dapi_rounds(),
        ANALYSIS_CONFIG,
    )
    assert len(result.ensemble_rows) == len(SIGMAS) * len(THRESHOLDS) * 2
    counts = [
        row.value
        for row in result.ensemble_rows
        if row.acquisition_round == 3 and row.endpoint == "component_count"
    ]
    assert np.median(counts) > 1
```

- [ ] **Step 2: Run the test and verify the Golgi module is absent**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_golgi.py -q
```

Expected: FAIL because `cyclic_stability.golgi` is missing.

- [ ] **Step 3: Implement the fixed perinuclear continuous-image analysis**

Build one per-cell ROI from the consensus nuclear mask: outside the nucleus and
within `18.0` expanded-image micrometers of its surface. Intersect with voxels valid
in every registered round. For each configured physical sigma, calculate 3D gradient
magnitude after background correction and smoothing, then Pearson correlation of
reference and moving gradients inside the fixed ROI. Reject zero-variance comparisons
instead of returning a perfect score. Report each scale and their prespecified median.

- [ ] **Step 4: Implement Golgi ensemble geometry**

For each threshold/scale member, retain all target-positive components rather than
only the largest. Report positive volume, component count, component-volume Gini,
fragmentation as component count per positive cubic micrometer, compactness,
surface area-to-volume, target-to-nuclear-surface distance quantiles, Jaccard,
boundary distances, and resolution-tolerant surface Dice relative to acquisition
round 2. Store metric-versus-threshold curves and the median across all frozen
members. Produce registered target overlays and threshold-robustness diagnostics.

Define compactness as `36 * pi * volume**2 / surface_area**3`, equal to one for an
ideal sphere, and compute component-volume Gini from the sorted positive component
volumes with the standard finite-sample formula. Record zero-volume masks as zero
signal/volume with NaN derived geometry rather than assigning favorable compactness
or overlap.

- [ ] **Step 5: Run tests and commit**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_golgi.py tests/test_morphology.py -q
git add scripts/cyclic_stability/golgi.py tests/synthetic.py tests/test_golgi.py
git commit -m "feat: measure golgi morphology stability"
```

---

### Task 7: LAMP1 gating, 3D detection, and resolution-aware matching

**Files:**

- Create: `scripts/cyclic_stability/lysosome.py`
- Modify: `tests/synthetic.py`
- Test: `tests/test_lysosome.py`

**Interfaces:**

- Consumes image-verified LAMP1 `CellSpec` values only, registered LAMP1/DAPI
  volumes, and the configured physical LoG scales and SBR thresholds.
- Produces
  `detect_lysosomes(image, roi, spacing_xyz_um, sigma_values_um, detection_sbr) -> tuple[DetectionSet, ...]`.
- Produces
  `match_lysosomes(reference: DetectionSet, moving: DetectionSet, resolution: ResolutionReference, max_units: float) -> ObjectMatchResult`.
- Produces
  `analyze_lamp1_cell(cell: CellSpec, artifacts, config, output_dir) -> MarkerResult`.
- Produces
  `eligible_lamp_cells(config: AnalysisConfig) -> tuple[CellSpec, ...]`.
- Production entry points report LAMP1 as `unavailable` rather than `passed` or
  `stable` while no eligible image-verified cells exist.
- Extend `tests/synthetic.py` with `synthetic_lamp_spots()`,
  `reference_detections()`, and `moved_lost_new_detections()`. The last two return
  exact `DetectionSet` values containing three reference objects and two retained
  objects plus one new object. Construct `RESOLUTION` through Task 3 and use
  `SPACING = (0.1, 0.1, 0.3)` in the test module.

- [ ] **Step 1: Write failing eligibility, detection, and matching tests**

```python
def test_unverified_lamp_inventory_is_not_schedulable(config_files: tuple[Path, ...]) -> None:
    config_without_lamp = load_analysis_config(*config_files)
    assert eligible_lamp_cells(config_without_lamp) == ()


def test_log_detector_recovers_anisotropic_physical_spots() -> None:
    detected = detect_lysosomes(
        synthetic_lamp_spots(), synthetic_roi(), SPACING, (0.35, 0.70, 1.40), (3.0, 4.0, 5.0)
    )
    assert all(item.object_count == 3 for item in detected)


def test_matching_allows_lost_and_new_objects_without_forcing_pairs() -> None:
    result = match_lysosomes(reference_detections(), moved_lost_new_detections(), RESOLUTION, 1.0)
    assert result.matched_count == 2
    assert result.lost_count == 1
    assert result.new_count == 1
```

- [ ] **Step 2: Run the test and verify the LAMP1 module is absent**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_lysosome.py -q
```

Expected: FAIL because `cyclic_stability.lysosome` is missing.

- [ ] **Step 3: Implement physical 3D LoG detection and object summaries**

Run scale-normalized negative Laplacian-of-Gaussian responses at physical sigma
`0.35`, `0.70`, and `1.40` micrometers after robust background subtraction. Detect
26-neighborhood maxima above each configured SBR threshold, suppress duplicate
maxima in anisotropic resolution coordinates, and watershed positive support around
retained peaks. Report count, volume, equivalent diameter, sphericity,
nearest-neighbor distance, and nucleus-relative distance for every ensemble member.
Define equivalent diameter as `(6 * volume / pi)**(1 / 3)` and sphericity as
`pi**(1 / 3) * (6 * volume)**(2 / 3) / surface_area`.

- [ ] **Step 4: Implement one-to-one anisotropic matching**

Build a cost matrix from normalized XYZ displacement:

```python
delta = moving_xyz[:, None, :] - reference_xyz[None, :, :]
cost = np.sqrt(
    (delta[..., 0] / resolution.lateral_expanded_um) ** 2
    + (delta[..., 1] / resolution.lateral_expanded_um) ** 2
    + (delta[..., 2] / resolution.axial_expanded_um) ** 2
)
```

Set costs above `1.0` to infinity and use `scipy.optimize.linear_sum_assignment` on
finite candidates. Do not force unmatched assignments. Report matched fraction,
lateral and axial displacement, normalized displacement, and lost/new fractions
relative to acquisition round 2 and adjacent rounds.

- [ ] **Step 5: Run tests, verify the real-data gate, and commit**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_lysosome.py -q
uv run --script scripts/validate_inputs.py \
  --analysis-config config/analysis.yaml \
  --correspondence config/correspondence.csv \
  --qc-decisions config/qc-decisions.csv \
  --cohort lamp1 --run-dir results/lamp-gate
```

Expected: unit tests pass; the current real-data command exits successfully with an
explicit `unavailable: no image-verified LAMP1 cell mappings` status and schedules
zero cells.

```bash
git add scripts/cyclic_stability/lysosome.py tests/synthetic.py tests/test_lysosome.py
git commit -m "feat: add gated lysosome stability analysis"
```

---

### Task 8: Atomic marker-cell tasks and bounded execution

**Files:**

- Create: `scripts/run_analysis.py`
- Create: `scripts/cyclic_stability/task.py`
- Create: `scripts/cyclic_stability/runner.py`
- Modify: `tests/synthetic.py`
- Test: `tests/test_task.py`
- Test: `tests/test_runner.py`

**Interfaces:**

- Consumes marker analyzers from Tasks 5--7, signal summaries from Task 4,
  registration QC from Task 3, and accepted `QcDecision` values from Task 1.
- Produces
  `build_tasks(config: AnalysisConfig, cohorts: tuple[str, ...]) -> tuple[CellTask, ...]`.
- Produces
  `run_cell_task(task: CellTask, run_dir: Path) -> TaskRecord`.
- Produces
  `run_tasks(tasks, profile: ExecutionProfile, run_dir: Path) -> RunRecord`.
- `CellTask.key` is exactly `{cohort}-{cell_id}`, and each task owns all five rounds.
- A task completion record contains config digest, ordered input identities, software
  versions, attempt directory, output identities, status, timestamps, peak RSS,
  elapsed seconds, process read bytes, and scratch bytes.
- Extend `tests/synthetic.py` with `cell_task(tmp_path)`,
  `synthetic_tasks(tmp_path)`, `profile(mode, max_workers=1)`, and
  `normalized_table_bytes(record)`. It also provides `different_identity()` as a
  valid `FileIdentity` unequal to the task's last input. These fixtures use a
  top-level picklable fake marker analyzer so the process test exercises a real
  spawned process rather than a mocked executor.

- [ ] **Step 1: Write failing restart and process-equivalence tests**

```python
def test_completed_task_is_reused_only_for_matching_inputs(tmp_path: Path, cell_task: CellTask) -> None:
    first = run_cell_task(cell_task, tmp_path)
    second = run_cell_task(cell_task, tmp_path)
    assert first.attempt_id == second.attempt_id
    assert second.reused is True
    changed = replace(cell_task, config_digest="changed")
    third = run_cell_task(changed, tmp_path)
    assert third.attempt_id != first.attempt_id
    assert third.reused is False


def test_changed_input_identity_invalidates_task_reuse(tmp_path: Path, cell_task: CellTask) -> None:
    first = run_cell_task(cell_task, tmp_path)
    changed = replace(
        cell_task,
        input_identities=(*cell_task.input_identities[:-1], different_identity()),
    )
    second = run_cell_task(changed, tmp_path)
    assert second.attempt_id != first.attempt_id
    assert second.reused is False


def test_sequential_and_process_modes_produce_identical_compact_tables(
    tmp_path: Path, synthetic_tasks: tuple[CellTask, ...]
) -> None:
    sequential = run_tasks(synthetic_tasks, profile("sequential"), tmp_path / "sequential")
    processes = run_tasks(synthetic_tasks, profile("process", max_workers=2), tmp_path / "process")
    assert normalized_table_bytes(sequential) == normalized_table_bytes(processes)
```

- [ ] **Step 2: Run the tests and verify orchestration is missing**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_task.py tests/test_runner.py -q
```

Expected: FAIL because task and runner modules are absent.

- [ ] **Step 3: Implement atomic attempts and exact reuse**

Before importing NumPy, ANTs, or BLAS-backed libraries, set
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
`BLIS_NUM_THREADS` from the validated profile; default each to `1`.

Write each attempt to `tasks/{key}/attempts/{uuid}/`. Flush and validate its CSV,
JSON, and diagnostic outputs, then atomically write
`tasks/{key}/complete.json` pointing to the immutable attempt. Exact reuse requires
the same config digest and ordered `FileIdentity` values for every source and Cy-ExM
artifact. Preserve failed attempts with `failure.json`; never let them authorize
reuse.

At run creation, copy the exact analysis YAML, correspondence CSV, QC-decision CSV,
and execution profile into `run_dir/config/`; record their SHA-256 digests and the
pinned Cy-ExM commit in `run.json`. When the CLI creates a run beneath a results
root, atomically write its absolute path to `results/latest-run.txt` after run
initialization.

`run_cell_task` dispatches by cohort, computes target-blind QC references, marker
morphology, and raw signal, then stores only within-cell tables and diagnostics. It
must reject missing QC decisions before loading a target image.

- [ ] **Step 4: Implement sequential, process, and SLURM-index modes**

Use `concurrent.futures.ProcessPoolExecutor(max_workers=profile.max_workers)` for
local process mode. Submit one `CellTask` per future and collect results in sorted
task-key order. Do not construct a second process pool inside a worker.

SLURM mode does not submit jobs from Python. `run_analysis.py --task-index N` runs
exactly `build_tasks(...)[N]`; README supplies the `sbatch --array` command derived
from the validated profile. A failed cell leaves completed cells reusable.

Wrap every task with `psutil.Process` resource sampling and record wall time,
`memory_info().rss` high-water samples, `io_counters().read_bytes` delta, and scratch
directory size. Do not interpret these values as scientific outcomes.

- [ ] **Step 5: Add the PEP 723 CLI and run focused tests**

`run_analysis.py` accepts `--analysis-config`, `--correspondence`, `--qc-decisions`,
`--execution-profile`, exactly one of `--run-dir` or `--results-root`, repeatable
`--cohort`, optional `--cell`, optional `--task-index`, and `--force-task`. With
`--results-root`, derive the run ID as UTC timestamp plus the first 12 config-digest
characters and update `latest-run.txt`. Its PEP 723 block contains runtime
dependencies but not pytest. Add `--pilot-first-accepted` as a mutually exclusive
alternative to `--cell`; it selects and records the lexicographically first cell in
the requested cohort whose moving rounds all have `accept` decisions.

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_task.py tests/test_runner.py -q
```

Expected: PASS, including interrupted-attempt recovery and byte-identical sequential
versus process compact tables.

- [ ] **Step 6: Commit orchestration**

```bash
git add scripts/run_analysis.py scripts/cyclic_stability/task.py scripts/cyclic_stability/runner.py tests/synthetic.py tests/test_task.py tests/test_runner.py
git commit -m "feat: run restart-safe marker-cell analyses"
```

---

### Task 9: Cell-level aggregation, editable figures, and Markdown report

**Files:**

- Create: `scripts/render_report.py`
- Create: `scripts/cyclic_stability/aggregate.py`
- Create: `scripts/cyclic_stability/figures.py`
- Create: `scripts/cyclic_stability/report.py`
- Create: `scripts/cyclic_stability/templates/report.md.j2`
- Modify: `tests/synthetic.py`
- Test: `tests/test_aggregate.py`
- Test: `tests/test_figures.py`
- Test: `tests/test_report.py`

**Interfaces:**

- Consumes only accepted task records, compact task tables, QC decisions, and
  selected diagnostic images; it never reopens full-resolution volumes.
- Produces
  `aggregate_run(run_dir: Path, config: AnalysisConfig) -> AggregateTables`.
- Produces
  `render_figures(tables: AggregateTables, run_dir: Path, config: AnalysisConfig) -> tuple[Path, ...]`.
- Produces
  `render_report(tables: AggregateTables, figures: tuple[Path, ...], run_dir: Path, config: AnalysisConfig) -> Path`.
- Every plotted point has cohort, cell, acquisition round, endpoint, units,
  comparison set, QC inclusion, and source task identity in a tidy table.
- Extend `tests/synthetic.py` with `synthetic_run(tmp_path) -> Path`, which writes
  five cell-level task records per marker, multiple objects per LAMP1 cell, accepted
  and excluded sensitivity rows, and exact compact-table identities. Define
  `ANALYSIS_CONFIG` from the frozen Task 1 fixture and implement
  `render_synthetic_report(path)` as a test helper that calls the production
  aggregator and renderer. Define `save_test_vector_figure(tmp_path, style)` in
  `tests/test_figures.py`; within `matplotlib.rc_context(style)` it saves the same
  one-line figure as PDF, PS, and SVG and returns a value with `pdf`, `ps`, and `svg`
  paths.

- [ ] **Step 1: Write failing replicate, style, and wording tests**

```python
def test_aggregation_uses_cells_not_objects_as_replicates(synthetic_run: Path) -> None:
    tables = aggregate_run(synthetic_run, ANALYSIS_CONFIG)
    row = tables.group_summary.query("endpoint == 'lamp1_object_count'").iloc[0]
    assert row["n_cells"] == 5
    assert row["n_objects"] > row["n_cells"]


def test_figure_style_preserves_editable_text() -> None:
    style = publication_rcparams()
    assert style["pdf.fonttype"] == 42
    assert style["ps.fonttype"] == 42
    assert style["svg.fonttype"] == "none"
    assert style["font.family"] == ["DejaVu Sans"]


def test_saved_vector_outputs_embed_truetype_and_keep_svg_text(tmp_path: Path) -> None:
    outputs = save_test_vector_figure(tmp_path, publication_rcparams())
    assert b"/FontFile2" in outputs.pdf.read_bytes()
    assert b"/FontType 42" in outputs.ps.read_bytes()
    assert b"<text" in outputs.svg.read_bytes()


def test_report_labels_theoretical_and_nominal_assumptions(synthetic_run: Path) -> None:
    report = render_synthetic_report(synthetic_run)
    assert "theoretical diffraction estimate" in report
    assert "nominal 4.5x expansion" in report
    assert "nonsignificant results demonstrate stability" not in report.lower()
    assert "failure to reject a round effect is not stability evidence" in report.lower()
```

- [ ] **Step 2: Run tests and verify aggregation/rendering is missing**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_aggregate.py tests/test_figures.py tests/test_report.py -q
```

Expected: FAIL because the aggregation and rendering modules are absent.

- [ ] **Step 3: Implement QC-aware cell-level tables**

Aggregate every within-cell distribution to prespecified quantiles and coverage
before cohort summaries. For each endpoint, create absolute round values, change
from acquisition round 2, adjacent-round change, and a descriptive per-cell slope
from rounds 1--5. Report median, range, and bootstrap percentile interval across
cells, with deterministic seed derived from config digest. Do not fit a complex
mixed-effects model or call a failure to reject a round effect equivalence.

Primary tables contain only the endpoints declared in the design specification.
Secondary tables retain volume, curvature, compactness, Jaccard, fragmentation,
and spatial distributions. Create separate primary-QC and all-data sensitivity
tables, plus separate rounds-2--5 signal and round-1 exposure-sensitivity tables.

- [ ] **Step 4: Implement publication figures with centralized styling**

Use this exact base style:

```python
def publication_rcparams() -> dict[str, object]:
    return {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": ["DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.transparent": True,
    }


CELL_PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
```

Render all-cell trajectories with consistent cell colors, thin individual lines,
cohort median overlays, and expanded/native theoretical reference annotations where
units permit. Render transform/Jacobian/seam QC, morphology endpoints, matched signal
rounds, round-1 sensitivity, ensemble robustness, representative microscopy panels,
and exclusions. Representative panels are explicitly labeled as 60x views and
include expanded-image scale bars plus the nominal 4.5x native-space equivalent.
Save each as PDF, SVG, and 600-dpi PNG. Keep microscopy pixels rasterized while
labels, scale bars, axes, and annotations remain vector.

- [ ] **Step 5: Implement the report and PEP 723 entry point**

The Jinja2 template contains: data provenance, acquisition numbering, QC/exclusions,
theoretical resolution, nominal expansion, deformation required for alignment,
post-registration morphology, repeated-labeling signal, threshold/scale sensitivity,
LAMP1 availability, limitations, Methods, Results, and concise rebuttal-ready text.
The wording states that signal reflects epitope accessibility plus staining and
imaging reproducibility and that transform-derived deformation includes model error.

`render_report.py` accepts `--run-dir` and `--analysis-config`, verifies every task
completion identity, writes tidy CSVs atomically, then renders figures and report
without accessing source image trees.

- [ ] **Step 6: Run focused tests, inspect outputs, and commit**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_aggregate.py tests/test_figures.py tests/test_report.py -q
git add scripts/render_report.py scripts/cyclic_stability/aggregate.py scripts/cyclic_stability/figures.py scripts/cyclic_stability/report.py scripts/cyclic_stability/templates/report.md.j2 tests/synthetic.py tests/test_aggregate.py tests/test_figures.py tests/test_report.py
git commit -m "feat: render cyclic stability evidence"
```

Expected: tests pass; synthetic PDF/PS settings are type 42, SVG output contains
editable `<text>` nodes, every trajectory shows all cells, and the report labels all
resolution and expansion assumptions.

---

### Task 10: Cropped-data smoke, resource pilot, production execution, and final gates

**Files:**

- Modify: `README.md`
- Create: `config/execution-local-pilot.yaml`
- Create: `scripts/make_smoke_data.py`
- Modify: `tests/synthetic.py`
- Create: `tests/test_smoke_cli.py`
- Runtime create: `smoke-data/`
- Runtime create: `results/`

**Interfaces:**

- Consumes all prior entry points and accepted target-blind QC decisions.
- Produces one cropped smoke dataset containing no source replacements and one
  immutable full-resolution run directory.
- Produces the final `run.json`, compact tables, figures, and Markdown report required
  by the design acceptance criteria.
- Extend `tests/synthetic.py` with
  `build_synthetic_five_round_fixture(tmp_path)`, `run_cli_fixture(fixture, mode, max_workers=1)`,
  `compact_table_digest(run_dir)`, and `expected_figure_triplets(run_dir)`. The
  fixture contains Golgi, Lamin, and image-verified LAMP1 cells, valid synthetic
  completion records, explicit accepted QC decisions, and small ZYX arrays.

- [ ] **Step 1: Write the failing end-to-end CLI smoke test**

```python
def test_synthetic_cli_run_is_restart_safe_and_renderable(tmp_path: Path) -> None:
    fixture = build_synthetic_five_round_fixture(tmp_path)
    first = run_cli_fixture(fixture, mode="sequential")
    second = run_cli_fixture(fixture, mode="process", max_workers=2)
    assert first.returncode == 0
    assert second.returncode == 0
    assert compact_table_digest(first.run_dir) == compact_table_digest(second.run_dir)
    assert (second.run_dir / "report.md").is_file()
    assert expected_figure_triplets(second.run_dir)
```

- [ ] **Step 2: Run the full synthetic suite before real cropped data**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py -q
```

Expected: all unit and synthetic end-to-end tests pass with no source or Cy-ExM
artifact writes.

- [ ] **Step 3: Implement and run a read-only cropped-data smoke**

`make_smoke_data.py` has the full runtime PEP 723 block, accepts one Golgi and one
Lamin cell, first runs the full ND2/export and Cy-ExM completion validation against
the original read-only inputs, then reads acquisition rounds 2 and 3 and writes
copies under `smoke-data/`. Choose a deterministic registered-grid crop centered on
the largest acquisition-round-2 DAPI component: all Z planes spanning that component
plus eight planes of padding, and a 512-by-512 XY window centered on its centroid and
clipped to image bounds. Apply the identical registered-grid crop to registered
DAPI/tubulin/target and warp arrays. For unregistered signal inputs, center the same
physical crop size on each round's raw DAPI component and record separate raw bounds.

Use Cy-ExM's `ArtifactStore.commit` to create test-only completion records inside
`smoke-data/`; their requests include the original validated identities and crop
bounds. Never copy those records into a Cy-ExM output tree. Run QC, analysis, and
rendering sequentially on the crop with fixture-only accepted QC decisions created
before target analysis. Inspect TIFF/ND2 reading, axis conventions, physical values,
memory behavior, report layout, font editability, and diagnostic panels.

- [ ] **Step 4: Run one full marker-cell resource pilot**

After the real target-blind `config/qc-decisions.csv` is complete, run one accepted
Lamin cell sequentially at full resolution. Record peak RSS, elapsed time, process
read bytes, and scratch bytes in its task record. Calculate the local worker count as:

```python
memory_workers = floor(0.70 * available_memory_bytes / pilot_peak_rss_bytes)
production_workers = max(1, min(3, memory_workers, remaining_task_count))
```

Keep the count at one if the pilot reports archive-read saturation, swap use, or
scratch exhaustion. Record the selected count and rationale in a new
`config/execution-local-pilot.yaml`; do not modify `config/analysis.yaml`.

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_analysis.py \
  --analysis-config config/analysis.yaml \
  --correspondence config/correspondence.csv \
  --qc-decisions config/qc-decisions.csv \
  --execution-profile config/execution-local.yaml \
  --results-root results/pilot \
  --cohort lamin --pilot-first-accepted
```

`--pilot-first-accepted` runs exactly the lexicographically first Lamin cell whose
four moving rounds all have accepted QC decisions and records the selected cell.

- [ ] **Step 5: Run current Golgi and Lamin production tasks restart-safely**

Create a run ID as UTC timestamp plus the first 12 characters of the scientific
config digest. Run all five Golgi and five Lamin cells with the pilot-derived local
profile. If local resource evidence is unacceptable, submit the exact ten tasks as a
SLURM array using the validated optional profile and `--task-index`.

On failure, preserve the failed attempt and rerun the same command so only missing or
stale cell tasks execute. Do not add LAMP1 tasks until the correspondence and
registration input gate accepts five image-verified cells.

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_analysis.py \
  --analysis-config config/analysis.yaml \
  --correspondence config/correspondence.csv \
  --qc-decisions config/qc-decisions.csv \
  --execution-profile config/execution-local-pilot.yaml \
  --results-root results \
  --cohort golgi --cohort lamin
```

- [ ] **Step 6: Render, visually review, and verify acceptance criteria**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
run_dir="$(sed -n '1p' results/latest-run.txt)"
test -d "$run_dir"
uv run --script scripts/render_report.py \
  --analysis-config config/analysis.yaml \
  --run-dir "$run_dir"
pdf_count="$(find "$run_dir/figures" -maxdepth 1 -name '*.pdf' -type f | wc -l)"
svg_count="$(find "$run_dir/figures" -maxdepth 1 -name '*.svg' -type f | wc -l)"
test "$pdf_count" -gt 0
test "$pdf_count" -eq "$svg_count"
rg -a -l '/FontFile2' "$run_dir"/figures/*.pdf >/dev/null
rg -a -l '<text' "$run_dir"/figures/*.svg >/dev/null
```

Verify:

1. all included sources and Cy-ExM artifacts match recorded identities;
2. acquisition round 2 is the reference in every table and figure;
3. deformation and residual morphology remain separate;
4. primary endpoints use cell-level replication;
5. rounds 2--5 and round-1 signal sensitivity remain separate;
6. QC-excluded and all-data sensitivity tables are both present;
7. LAMP1 is clearly unavailable or fully image-mapped and analyzed;
8. all ensemble members and robust medians are recoverable from tidy tables;
9. each PDF/SVG/PNG figure triplet exists and passes final-size visual review;
10. PDF and PS text is type 42 and SVG text remains editable;
11. expanded-image and nominal 4.5x native-space labels are explicit; and
12. report prose avoids unsupported stability, epitope-chemistry, and measured-PSF
    claims.

- [ ] **Step 7: Run final code gates and commit the implementation record**

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py -q
uvx ruff check scripts tests
uvx ruff format --check scripts tests
git status --short
```

Expected: all tests pass, Ruff reports no errors or formatting changes, generated
data remain ignored, and only the named README, smoke script, and smoke test are
staged for the final source commit.

```bash
git add README.md config/execution-local-pilot.yaml scripts/make_smoke_data.py tests/synthetic.py tests/test_smoke_cli.py
git commit -m "test: validate cyclic stability workflow end to end"
```

The final handoff records the external repository commit, immutable analysis run ID,
test count, resource pilot, included/excluded cell-rounds, LAMP1 availability,
figure inventory, and the absolute path to `report.md`.
