# Design and implementation artifact guidance

- Store dated, reviewable design specifications under `specs/` and executable
  implementation plans under `plans/`; use `YYYY-MM-DD-<topic>.md` names.
- A design records approved decisions and boundaries. Its implementation plan is a
  separate artifact written only after the design is reviewed.
- Keep completed Phase 1 artifacts as historical evidence. Record a correction in a
  new dated document or an explicitly labeled erratum instead of silently rewriting
  history.
- Do not commit generated review packages, temporary evidence, agent ledgers, command
  transcripts, or files from `.superpowers/sdd`.
- Link maintained user documentation to current behavior; design and plan documents
  may describe proposed work only when they label it clearly.
