# Notebook example guidance

- The first code cell is an obvious `# EDIT THESE PATHS` block using `Path` variables
  and portable placeholder values. Put run parameters beside those paths when that
  makes the notebook easier to reproduce.
- Clear every code cell before committing: `outputs` is empty and `execution_count`
  is `null`. Avoid shell commands, notebook magics, hidden state, and downloads.
- Keep every code cell valid Python so examples can be inspected and compiled without
  launching Jupyter. Configure native threads before importing or loading scientific
  runtimes.
- Replace legacy plotting, statistics, and file-operation helpers with small local
  NumPy and Matplotlib operations only where they clarify registration behavior.
- Explain transform ownership, interpolation, XYZ spacing versus ZYX arrays, and the
  exact artifact filenames used by each workflow.
- Never preserve machine-specific paths, generated images, large archives, or stale
  API calls merely because they appeared in the source notebook.
