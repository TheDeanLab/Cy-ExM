# Implementation-plan guidance

- Name exact files, interfaces, dependency order, commands, expected results, review
  gates, and commit boundaries. An implementer must not need unstated repository
  knowledge.
- Break work into independently testable tasks with explicit TDD red and green steps.
  Show complete test and implementation snippets rather than placeholders.
- Use checkboxes as execution state, not prediction. Do not mark a step complete until
  fresh evidence from its exact command exists.
- Include focused tests before broad gates and preserve repository safety constraints
  in every task that can touch external data, native runtimes, or GitHub state.
- Completed historical plans remain evidence of the path actually taken. Use a new
  correction plan when scope or implementation changes materially; do not rewrite
  historical plans to conceal that change.
