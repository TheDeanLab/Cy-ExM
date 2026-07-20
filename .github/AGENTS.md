# GitHub automation guidance

- Keep workflow and repository permissions least-privilege; ordinary CI needs only
  read access to repository contents.
- Use immutable commit pins or explicit version-pinned releases for third-party
  actions. Verify a requested upgrade exists and review its release notes before
  changing the pin.
- Keep portable validation on public GitHub-hosted runners. Never embed credentials,
  private tokens, lab paths, `/project` or `/archive` locations, or cluster account
  details in committed automation.
- Do not make release publication, branch-protection changes, repository visibility
  changes, or destructive maintenance part of an unrelated implementation task.
- Workflow changes must update their exact repository-policy assertions and pass YAML
  parsing plus the affected local commands before review.
