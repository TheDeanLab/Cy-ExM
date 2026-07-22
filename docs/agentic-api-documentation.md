# Agentic API documentation guide

Use this guide when adding or changing a user-callable Cy-ExM API. It makes existing
behavior, extension points, ownership boundaries, and safe alternatives discoverable
for users and coding agents.

## Scope and decision order

A user-callable API is a relatively high-level entry point, resource owner, data
model, executor, or explicitly supported extension interface. It is not every
non-private Python symbol or every module-level `__all__` member. Full Numpydoc is
required for user-callable APIs; concise Numpydoc is sufficient for simple pure value
objects and helpers whose behavior is clear from their typed signature.

Before adding a class, function, method, property, field, adapter, protocol, or
helper:

1. Search source, tests, documentation, examples, public exports, the proposed name,
   and plausible synonyms.
2. Inspect relevant docstrings, especially `See Also`, `Notes`, `Attributes`, and
   `Examples`.
3. Identify the closest existing symbols and decide whether composition, extension,
   parameterization, or repair represents the requested behavior.
4. Prefer improving the existing abstraction. Do not add a wrapper that only renames
   an operation, forwards the same arguments, or reproduces its lifecycle.
5. Do not independently reimplement orchestration, validation, normalization,
   persistence, state management, or cleanup already owned by an existing symbol.
6. Add a new symbol only when existing APIs cannot be safely extended or composed
   without an incorrect abstraction or a compatibility break.

Treat documented units, array shapes, coordinate systems, state transitions, side
effects, ownership, and expected errors as behavioral contracts. Anything intended
for reuse must be explicitly designated and documented; private implementation
details are not implicit extension interfaces.

## Discovery commands

Search definitions, exports, protocols, and documentation before designing a new API:

```bash
rg -n '^(class|def) |^\s+(def|async def) ' src
rg -n '__all__|Protocol|ABC|@overload|@property' src
rg -n '^\s*(See Also|Notes|Attributes|Methods|Examples)$' src
rg -n 'desired_domain_term|known_synonym' src tests docs examples
fd -t f . src tests docs examples
```

For workflow work, begin with `build_workflow_plan`, `run_workflow`,
`inspect_workflow_status`, manifest loaders, and executor classes. For Phase 1
registration, begin with `ImageRegistration`, `ChunkedImageRegistration`,
`register_round`, and `transform_channel`. These are starting points, not permission
to bypass lower-level contracts.

## Public API design

- Use task-oriented names that distinguish high-level orchestration from lower-level
  primitives. Avoid multiple public synonyms for the same operation.
- Use complete type annotations and domain-specific return types. Prefer keyword-only
  parameters for optional behavior with semantic meaning.
- Use dataclasses, enums, protocols, abstract base classes, and overloads only when
  they clarify an extension boundary or accepted behavior.
- Maintain a deliberate `__all__` for a supported public module API, but do not export
  implementation details merely because another internal component needs access.
- For every important symbol, document what it owns and does not own, its canonical
  entry points, adjacent responsibility, and supported composition or extension path.

## Numpydoc contract

Use standard Numpydoc section names; do not introduce custom sections. Every
user-callable API includes a direct one-line summary and, when behavior is nontrivial,
an extended summary. Document `Parameters`, `Returns` or `Yields`, and `Raises` when
applicable. Important entry points, resource owners, data models, executors, and
extension interfaces also include `See Also`, `Notes`, and canonical `Examples`.

The extended summary states whether the symbol is the canonical entry point, when to
use it rather than a lower-level or specialized path, and which responsibility it
does not own. It should not be an implementation walkthrough.

### Parameters

Type annotations are the source of truth for Python types; parameter descriptions add
semantics that types cannot express. Document applicable units, axes, dtype limits,
`None` semantics, defaults, copying or mutation, path existence or creation,
coordinate systems, ownership, ordering, and internal normalization.

For image APIs, preserve Cy-ExM's scientific conventions: public pixel spacing is in
micrometers in `(x, y, z)` order; NumPy volumes use `(z, y, x)` order; fixed and
moving roles do not change. State when an operation crosses the fixed-grid boundary
or consumes a persisted transform chain.

```text
moving : ArrayLike
    Moving volume with shape ``(z, y, x)``. The input is not mutated. Its native
    geometry may differ from the fixed image; affine application establishes the
    fixed grid before nonlinear processing.
```

### Returns, yields, and raises

Name nontrivial results and document semantic meaning, shape and dtype, ownership,
mutability, view-versus-copy behavior, persistence state, and relation to inputs when
applicable. Document caller-visible exception conditions, not incidental implementation
errors.

### See also

Treat `See Also` as the API reuse graph. Include the high-level path normally
preferred, lower-level composed primitives, sibling alternatives, relevant data
models, inverse operations, and genuine extension points. Each description states the
decision boundary instead of repeating the name.

```text
See Also
--------
build_workflow_plan :
    Validate manifests and create deterministic work without executing rounds.
run_workflow :
    Execute pending planned rounds and persist operational provenance.
inspect_workflow_status :
    Inspect authoritative completion state without changing workflow state.
```

### Notes, attributes, and examples

Use `Notes` for prerequisites and transitions; mutation and side effects; idempotency;
thread, process, and async safety; ownership; cache invalidation; persistence; and
composition that extensions must preserve. Document public attributes behaviorally:
their source of truth, initialization, mutability, valid states, units, shape, axes,
`None` meaning, and direct-assignment support.

Examples demonstrate intended architecture, not a reimplementation of internals. For
important APIs, include the canonical high-level path and, when supported, a
customization path that composes existing primitives. Keep examples synchronized with
actual signatures; do not contact a scheduler, depend on lab storage, or imply
unaccepted scientific defaults.

## Class and module API maps

The opening paragraph of an important class docstring identifies its architectural
role, what it owns and does not own, its lifecycle, canonical entry points, and the
neighboring responsibility. Resource owners and executors must be especially clear
about resource lifecycle and cleanup.

Reusable-component module docstrings provide an API map: singular responsibility,
canonical entry point, public classes and functions, orchestration versus primitives,
and the neighboring module that handles adjacent work. Maintain `__all__` where it
defines a supported public module surface.

## Public API change synchronization

When user-callable behavior changes, update all applicable artifacts:

- Type annotations and the owning docstring.
- Related `See Also`, class `Attributes`, and module API-map entries.
- `__all__` when the supported module surface changes.
- Canonical examples, user-facing documentation, and applicable tests.
- Changelog or migration guidance when compatibility changes.

Keep scientific claims about transforms, interpolation, masks, axes, metrics,
schedules, and tolerances synchronized with accepted implementation and evidence.

## Reuse analysis records

For multi-step work or a user-callable API change, include this detailed record in the
implementation plan before implementation:

```markdown
## Reuse analysis

- Closest existing symbols: `build_workflow_plan`, `run_workflow`.
- Decision: extend `build_workflow_plan` with a keyword-only option.
- Rejected alternative: a separate planner would duplicate validation, ordering, and
  artifact-lifecycle rules.
- Documentation updates: API map, `See Also`, and canonical example.
```

The commit that adds or materially changes the API or extension boundary records the
final concise decision in its body:

```text
Reuse analysis:
- Extended `build_workflow_plan`; did not add a filtered-plan builder.
- A separate builder would duplicate validation, ordering, and artifact lifecycle.
- Updated the API map, See Also, and canonical example.
```

When a pull request is opened, its description links to the implementation plan and
relevant commit(s), then notes any implementation-time deviation. Do not duplicate the
detailed plan analysis in the PR. The commit record is mandatory for local work with
no PR. Preserve the concise reuse analysis in a squash commit. This record is not
required for a change that does not materially affect a user-callable API or extension
boundary.

## Numpydoc validation adoption

Numpydoc validation is not currently a configured repository gate. When a separately
approved tooling change adds it, begin with structural checks such as `GL08`, `SS01`,
`SS02`, `SS03`, `SS05`, `SS06`, `PR01`, `PR02`, `PR03`, `PR04`, `PR07`, `RT01`,
`RT02`, `RT03`, `YD01`, `SA02`, `SA03`, and `SA04`, excluding tests and trivial
`__repr__` and `__str__` methods. Apply `ES01`, `SA01`, and `EX01` only to the
important user-callable scope, not every helper.

If API documentation later uses Sphinx, configure Numpydoc class-member and
cross-reference behavior deliberately. Do not add generated API reference pages or
documentation dependencies solely as a side effect of a code change.

## Canonical Cy-ExM template

Use this structure for an important workflow entry point, adapting names and sections
to actual behavior rather than copying it mechanically:

```python
def run_workflow(...):
    """Execute pending workflow rounds and persist operational provenance.

    This is the canonical execution entry point after manifest and profile paths have
    been selected. It plans work deterministically, prepares the reference round,
    then executes moving rounds through the selected executor.

    Use `build_workflow_plan` when only validation and deterministic planning are
    needed. Use `inspect_workflow_status` to inspect completion without changing
    workflow state.

    Parameters
    ----------
    dataset_manifest_path : str or Path
        Scientific dataset manifest. It defines fixed and moving round roles,
        registration inputs, and public voxel spacing in ``(x, y, z)`` micrometers.

    See Also
    --------
    build_workflow_plan :
        Validate inputs and construct deterministic work without filesystem writes.
    inspect_workflow_status :
        Inspect authoritative completion provenance without executing work.

    Notes
    -----
    The reference round is a hard prerequisite. Moving rounds are not submitted when
    reference preparation fails. Round tasks preserve serial registration work and
    use the Phase 1 artifact store as the only restart authority.
    """
```

