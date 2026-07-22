# Jupyter Book GitHub Pages design

## Context

Cy-ExM currently has Markdown documentation in `docs/`, but no published
documentation site. The project needs a readable, narrative-oriented site at
`https://TheDeanLab.github.io/Cy-ExM/` that explains both usage and the scientific
and technical choices behind registration.

The repository's GitHub Pages source has been configured to **GitHub Actions**.

## Goals

- Publish a polished, static documentation site from the `main` branch.
- Keep documentation close to the existing Markdown source and use MyST Markdown
  for new pages.
- Prioritize tutorials, technical explanation, scientific invariants, and measured
  characterization evidence over generated API reference documentation.
- Present the brightness-dependence investigation as a transparent technical and
  scientific evidence chapter.
- Make the in-progress CC SyN radius comparison visible as ongoing work, without
  presenting its results as accepted evidence before characterization completes.
- Automatically build and publish when documentation sources change on `main`.

## Non-goals

- Do not publish raw datasets, generated registration outputs, large artifacts, or
  private lab-storage paths.
- Do not execute notebooks, download data, contact a scheduler, access lab storage,
  or run SLURM as part of documentation builds.
- Do not add generated Python API reference pages in the first release.
- Do not change registration behavior, scientific defaults, artifact semantics, or
  the package's base runtime dependencies.

## Alternatives considered

1. **Jupyter Book 2 with MyST Markdown and Book Theme (selected).** This is the
   actively developed, narrative-focused option. It preserves the existing Markdown
   source and supports scientific prose, figures, citations, math, and future
   notebooks.
2. **Sphinx with MyST Markdown and Sphinx Book Theme.** This retains the legacy
   Sphinx book stack but is in maintenance mode and does not offer a compelling
   advantage while API reference generation is explicitly deferred.
3. **Sphinx with MyST Markdown and PyData Sphinx Theme.** This is a strong
   software-documentation option, but its visual and organizational model is less
   aligned with a tutorial-and-evidence book.

## Selected architecture

Use Jupyter Book 2, authored in MyST Markdown and configured with a single
`myst.yml` file. The `docs/` directory remains the documentation source of truth.
The configuration defines site metadata, the GitHub project base URL, navigation,
and build settings. Rendered output is disposable CI build output and is never
committed.

Initial navigation is:

```text
Home
|- Getting started
|  |- Installation
|  `- First registration workflow
|- Scientific and technical foundations
|  |- Registration model and geometry
|  |- Artifact recovery and reproducibility
|  `- Workflow execution model
|- Tutorials
|  |- Whole-image registration
|  |- Serial chunked registration
|  `- Phase 2 workflow manifests and executors
`- Characterization and evidence
   |- Bright versus dim structural regions
   |- CC SyN radius characterization -- ongoing
   `- Phase 1 acceptance evidence
```

Existing maintained pages are reused or adapted into these sections. Historical
plans and specifications under `docs/superpowers/` are not site navigation content.

The brightness chapter explains the observation, method, results, interpretation,
limitations, and next experiment. It must distinguish accepted findings from
proposed targets. CC radius is a scalar voxel radius, and radius 4 remains
provisional until the approved characterization has established otherwise.

## Publishing data flow

1. A change is merged or pushed to `main` affecting documentation sources.
2. A dedicated Pages workflow creates the locked documentation build environment.
3. Jupyter Book validates and renders static files under a CI-only output directory.
4. The workflow uploads that directory with GitHub's Pages artifact action.
5. A dependent deployment job publishes the artifact to the `github-pages`
   environment at `https://TheDeanLab.github.io/Cy-ExM/`.

The workflow also supports `workflow_dispatch` for a manual re-publication. It is
separate from the software CI workflow and grants only the Pages deployment
permissions needed by the deployment job.

## Failure handling

- Invalid MyST, unresolved local references, or a failed build prevents artifact
  upload and deployment; the previous published site remains intact.
- A GitHub Pages deployment failure leaves the build artifact available in the
  workflow run for inspection and does not modify package or scientific outputs.
- Documentation claims about execution status distinguish portable construction
  evidence from real SLURM acceptance, following `docs/AGENTS.md`.

## Validation and acceptance criteria

- The documentation build succeeds locally using the locked project environment.
- The Pages workflow runs for a `main`-branch documentation change, uploads the
  static site, and deploys it successfully through GitHub Actions.
- The published site resolves its homepage, navigation, assets, and internal links
  beneath `/Cy-ExM/`.
- The first publication includes readable narrative pages for installation,
  registration concepts, workflow guidance, artifact recovery, Phase 1 evidence,
  and brightness-dependence characterization.
- The build remains data-free: it does not run notebooks or use lab storage,
  schedulers, or generated registration artifacts.
