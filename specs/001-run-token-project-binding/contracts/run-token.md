# Contract: run token injected into tools and active agents

**Feature**: `001-run-token-project-binding` | **Consumers**: user tools and active agents during `weni run`, then retail-setup (unchanged, FR-015)

## Unchanged

| Aspect | Value |
|---|---|
| Claims | `project_uuid`, `exp`, `iat`. Nothing added or removed (FR-006). |
| Algorithm and key | RS256 with `JWT_SECRET_KEY` |
| Lifetime | `DEFAULT_EXPIRATION_MINUTES = 2` |
| Injection point | `project.auth_token` (`JWT_PROJECT_KEY`), inside `sessionAttributes.project` for tools and the event's `project` for active agents |
| Frequency | One token per test case that needs one |
| Active pass-through | A test case whose `project` already has `auth_token` keeps it unchanged, with no mint (FR-007) |

## Changed

| Aspect | Before | After |
|---|---|---|
| Source of `project_uuid` | Body `project_uuid` (`str(data.project_uuid)`) | Canonical authorized header, `str(UUID(X-Project-Uuid))` (FR-004). Because the header must equal the raw body, the byte value conforming clients receive is the same as today. |
| Audit | None | One `run_token_minted` record per token ([log-events.md](./log-events.md)) |
| Documentation | Docstring says the default is 60 minutes | Docstring says 2 minutes (FR-008). No code change in `jwt_generator.py`. |
