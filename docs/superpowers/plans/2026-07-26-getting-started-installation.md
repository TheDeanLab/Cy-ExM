# Getting Started Installation Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the Getting Started installation guidance so it mirrors every supported README installation path in accessible, portable scientific-language documentation.

**Architecture:** Change only `docs/getting-started.md`; it remains a concise on-ramp and delegates workflow concepts to the existing workflow guide. Use commented `uv sync --locked` commands as the primary interface, grouped by the task a scientist wants to perform. Verify the rendered MyST/Jupyter Book page rather than adding an unnecessary documentation-specific test framework.

**Tech Stack:** Markdown, MyST/Jupyter Book 2.1.6, `uv`.

## Global Constraints

- Cy-ExM requires Python `>=3.12,<3.13`.
- Preserve the README's exact supported installation commands and the cumulative-extra relationship: `distributed` includes `workflow`; `hpc` includes `distributed`.
- Do not add dependencies, alter packaging, explain registration/execution modes, or prescribe Danuser-lab-specific `/project` or `/archive` paths.
- Describe HPC cache locations portably: users must follow their own cluster's quota, performance, and module guidance.
- Keep user documentation to behavior available on the current branch and retain the existing workflow-guide link.

---

### Task 1: Refresh the task-oriented installation page

**Files:**
- Modify: `docs/getting-started.md`

**Interfaces:**
- Consumes: the installation contract in `README.md` lines 17–60 and the approved design in `docs/superpowers/specs/2026-07-26-getting-started-installation-design.md`.
- Produces: one self-contained Getting Started installation section for scientists choosing a supported capability.

- [ ] **Step 1: Write the failing documentation-content check**

Run the following read-only check before editing. It encodes the missing user-facing capabilities that the updated page must contain:

```bash
rg -n --fixed-strings \
  -e "Notebooks and parameterized example scripts" \
  -e "--group dev --extra examples" \
  -e "according to your site's storage and module guidance" \
  docs/getting-started.md
```

Expected: fail because the current page lacks the notebooks/examples command, the development convenience command, and portable HPC guidance.

- [ ] **Step 2: Confirm the check is red**

Run:

```bash
rg -n --fixed-strings \
  -e "Notebooks and parameterized example scripts" \
  -e "--group dev --extra examples" \
  -e "according to your site's storage and module guidance" \
  docs/getting-started.md
```

Expected: nonzero exit status with no matches for the missing content.

- [ ] **Step 3: Replace the installation copy with the approved structure**

Edit `docs/getting-started.md` so it contains, in this order:

````markdown
Cy-ExM requires Python 3.12. Start with the locked base environment when you
want to use the Python registration API:

```bash
uv sync --locked
```

Add the optional capability that matches your task. Extras are cumulative:
`distributed` includes `workflow`, and `hpc` includes `distributed`.

```bash
# YAML manifests and sequential workflow execution
uv sync --locked --extra workflow

# Local process-based parallel execution with Dask
uv sync --locked --extra distributed

# SLURM/HPC execution through dask-jobqueue
uv sync --locked --extra hpc

# Notebooks and parameterized example scripts
uv sync --locked --extra examples
```
````

Then add concise paragraphs stating that `--no-dev` makes a lean environment;
that `uv sync` does not activate `.venv`, so commands use `uv run`; and that
the established example conveniences are:

```bash
# Lean runtime plus examples
uv sync --no-dev --extra examples

# Examples plus development and testing tools
uv sync --group dev --extra examples
```

Finish with portable HPC guidance using this exact sentence:

```markdown
On a quota-limited HPC system, configure uv's cache and managed Python location
according to your site's storage and module guidance; choose a location with
adequate quota and performance before creating the environment.
```

Retain the existing link to `workflow-guide.md`; do not add explanations of
whole-image, chunked, coarse, or fine-grained modes.

- [ ] **Step 4: Run the focused content check to verify it is green**

Run:

```bash
rg -n --fixed-strings \
  -e "Notebooks and parameterized example scripts" \
  -e "--group dev --extra examples" \
  -e "according to your site's storage and module guidance" \
  docs/getting-started.md
```

Expected: all three required strings are present.

- [ ] **Step 5: Inspect README parity and forbidden site-specific paths**

Run:

```bash
sed -n '17,60p' README.md
sed -n '1,220p' docs/getting-started.md
! rg -n '/project|/archive|coarse|fine_grained|whole_image|serial_chunked' \
  docs/getting-started.md
```

Expected: every README installation capability appears in the guide, the prose
is suitable for an educated scientific audience, and the final command exits
zero because the page contains neither lab-specific paths nor mode discussion.

- [ ] **Step 6: Build the user documentation**

Run:

```bash
cd docs && uv run --group docs jupyter book build --html
```

Expected: exit code 0 and a rendered `docs/_build/html/getting-started.html`.

- [ ] **Step 7: Commit the reviewed documentation change**

Run:

```bash
git add docs/getting-started.md
git commit -m "docs: expand getting started installation guidance"
```

Expected: one documentation-only commit. Do not stage generated `docs/_build/`
output or running-workflow SLURM logs.

## Plan self-review

- **Spec coverage:** Task 1 covers the base, workflow, distributed, HPC,
  examples, development, and lean installation paths; cumulative extras;
  portable HPC guidance; README parity; and a successful documentation build.
- **Scope:** The only user-facing file is `docs/getting-started.md`; no behavior,
  dependency, or workflow-mode material changes.
- **Ambiguity resolution:** The portable HPC sentence and the focused check use
  the identical wording from Step 3.
