# Design-specification guidance

- Separate current behavior from proposed behavior. Define context, goals, non-goals,
  alternatives, selected architecture, data flow, failure handling, testing, and
  acceptance criteria.
- State scientific invariants, public coordinate conventions, dependency boundaries,
  and ownership of persistent state wherever the design can affect registration.
- Label environment assumptions and verify them during implementation. Cluster
  partition, account, resource syntax, shared-filesystem, startup, and memory details
  are evidence requirements, not facts to invent.
- Resolve ambiguous public APIs and error behavior in the specification before an
  implementation plan is written. Avoid placeholders and open-ended requirements.
- Preserve completed historical specifications. Put a material correction in a new
  dated specification or add an explicitly labeled erratum that explains the change.
