# Agentic API documentation design

## Context

Cy-ExM's root contributor guidance now contains a detailed proposed Numpydoc and
API-reuse policy. The policy's intent is to make user-callable APIs and deliberate
extension surfaces easy for humans and coding agents to discover and reuse, rather
than encouraging parallel orchestration, validation, persistence, or lifecycle
implementations. Keeping the entire guide in the root `AGENTS.md` obscures the
scientific, runtime, and storage rules that apply to every task.

The current package has a deliberately narrow package-level API but also exposes
some module-level interfaces. Documentation requirements therefore need a semantic
definition of public API rather than treating every non-private symbol as stable.

## Goals

- Keep a concise, universal reuse rule in the root `AGENTS.md` so it is considered
  before an agent chooses a source subtree.
- Define a user-callable API as a relatively high-level operation, entry point,
  resource owner, data model, executor, or explicit extension interface.
- Require full Numpydoc for those important interfaces, while allowing concise
  Numpydoc for simple pure value objects and helpers.
- Place detailed section guidance, templates, and validation advice in one maintained
  reference document.
- Require a reviewable reuse decision before and after an API change: detailed in an
  implementation plan and concise in the API-changing commit; PR descriptions link
  to both instead of duplicating them.

## Non-goals

- Do not document every internal helper as a supported contract.
- Do not change package behavior, imports, scientific invariants, or dependencies.
- Do not add Numpydoc, Sphinx, doctest, CI, or pre-commit dependencies in this policy
  change. Validation configuration remains guidance until a separately approved
  adoption task selects and installs the tooling.
- Do not require a PR for local work or impose reuse prose on commits that do not add
  or materially change a user-callable API or extension boundary.

## Alternatives considered

1. **Keep the full policy in the root `AGENTS.md`.** This makes it universally
   visible but dilutes high-priority repository constraints and makes the root file
   costly to scan.
2. **Put all policy in child `AGENTS.md` files.** This keeps the root concise, but
   an agent may choose a directory or design a parallel abstraction before reading a
   child policy.
3. **Concise root rule plus detailed reference and scoped child rules (selected).**
   The root establishes the required decision order. A linked guide supplies the
   reusable Numpydoc material; source and documentation child guidance specify when
   it applies during code and documentation work.

## Selected architecture

`AGENTS.md` retains a short `Existing API reuse and documentation policy` section.
It requires discovery before new symbols, defines the reuse order, defines the
user-callable scope, distinguishes full and concise documentation, and points to
`docs/agentic-api-documentation.md` for the complete guide.

`docs/agentic-api-documentation.md` is the maintained reference for discovery
commands; Numpydoc section semantics; module API maps; `__all__`; canonical examples;
validation guidance; and the full documentation template. Its examples use Cy-ExM
terms and preserve its fixed/moving roles, public `(x, y, z)` spacing convention,
and external `(z, y, x)` NumPy volume convention.

`src/cyexm/AGENTS.md` specifies the classification rule used while modifying Python
code: full documentation applies to user-callable entry points, resource owners,
data models, executors, and explicitly supported extension interfaces. Simple pure
value objects and helpers may use concise Numpydoc. Any intentionally reusable
internal interface must be explicitly designated and documented rather than being
private by convention.

`docs/AGENTS.md` requires maintained API maps, `See Also` decision boundaries, and
canonical examples to follow the reference guide when public behavior changes.

For a multi-step or user-callable API change, the implementation plan contains a
`Reuse analysis` section naming the closest symbols, final decision, rejected
alternatives, and necessary documentation updates. The commit that adds or
materially changes that API repeats a concise final `Reuse analysis` in its body.
When a PR is opened, its description links to the plan and relevant commit(s), notes
any deviation, and does not duplicate the detailed analysis. The commit is the
durable local-work record; squash merges must preserve its reuse analysis.

## Data flow

1. An agent receives a requested code change and reads the concise root rule.
2. Before designing a user-callable API, it searches source, tests, docs, examples,
   exports, proposed names, and synonyms.
3. The agent records the pre-implementation reuse decision in the implementation
   plan when the change is multi-step or changes a user-callable API.
4. It reads the detailed guide and applicable child guidance, then extends or
   composes the closest existing symbol where safe.
5. The API-changing commit retains the concise final decision. A PR links the plan
   and commit record for review.

## Failure handling

- If no existing API safely represents the behavior, the reuse analysis documents
  why extension or composition would create an incorrect abstraction or break
  compatibility before a new symbol is introduced.
- If no PR is created, the API-changing commit remains the required durable record.
- If a squash commit is used, the author preserves the concise reuse analysis in the
  squash commit body.
- If a symbol is not user-callable and is not an explicit extension interface, it
  does not gain a broad stable contract solely to satisfy documentation policy.

## Validation and acceptance criteria

- Root `AGENTS.md` contains the concise universal rule and a link to the detailed
  guide, without retaining the full template and validation catalog.
- The detailed guide captures the current approved Numpydoc and reuse content in
  Cy-ExM terminology and clearly labels validation tooling as future adoption
  guidance.
- `src/cyexm/AGENTS.md` and `docs/AGENTS.md` apply the documented scope in their
  respective contexts.
- The policy requires `Reuse analysis` in plans and qualifying commits, and specifies
  PR linkage and squash-merge preservation.
- Markdown links and headings are valid under the repository's configured
  documentation build.
