# Data Model: Bind CLI Run Tokens to the Authorized Project

**Feature**: `001-run-token-project-binding` | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md)

The service has no database. These are the in-process values and the log
records the feature introduces or constrains. Nothing is persisted outside
application logs (Constitution XII: stateless).

---

## Authorized project

The project the user was authorized for.

| Field | Source | Rules |
|---|---|---|
| raw value | `X-Project-Uuid` request header | Present and authorized by `AuthorizationMiddleware` (Connect) before any dependency runs. This feature does not change it. |
| canonical value | `str(UUID(raw value))` | Used only as the run token's `project_uuid` claim. Always succeeds after the binding check (see Requested project). |

## Requested project

The top-level `project_uuid` of the request body on the four covered
endpoints.

| Field | Source | Rules |
|---|---|---|
| validated value | Body model field `project_uuid: UUID4` | A missing or invalid value produces today's 422. It is never reported as a mismatch. |
| raw value | Form field (runs, agents push) or JSON key (channels, ticketers) as sent | Compared with the authorized raw value as an **exact string**. Any difference, including letter case or UUID formatting, is a mismatch and yields `ProjectMismatchError`. |

Not covered: `VerifyPermissionRequestModel.project_uuid` (`/permissions/verify`)
and any project identifier nested inside `ticketer_definition` (FR-005).

## Attributed run

The value the run endpoint receives from `attributed_run_request` once both
checks have passed.

| Field | Type | Rules |
|---|---|---|
| `request` | `RunRequestModel` | Already bound to the authorized project. |
| `authorized_project_uuid` | `str` | Canonical authorized project. |
| `user_email` | `str` | Non-empty `email` claim of the bearer JWT, read without signature verification after Connect validated the token. |

**State flow per run request**:

```text
received
  -> header missing ............................ 400 (middleware, unchanged)
  -> header not authorized in Connect ......... 401/403 (middleware, unchanged)
  -> body invalid .............................. 422 (unchanged)
  -> body project != header (exact) ............ 403 PROJECT_MISMATCH + project_mismatch_rejected
  -> identity unreadable ....................... 403 RUN_NOT_ATTRIBUTABLE + run_not_attributable
  -> attributed ................................ endpoint runs as today (streaming)
```

Agents push, channel creation and ticketer creation follow the same flow
without the identity step.

## Run token issuer

`RunTokenIssuer` (`app/services/runs/token_issuer.py`) is created by the run
endpoint once per run and passed to the strategy in use.

| Field | Type | Source |
|---|---|---|
| `authorized_project_uuid` | `str` | Attributed run |
| `user_email` | `str` | Attributed run |
| `agent_key` | `str` | `RunRequestModel.agent_key` |
| `tool_key` | `str \| None` | `RunRequestModel.tool_key` (passive runs only; `None` for active runs) |
| `run_type` | `"passive" \| "active"` | `RunRequestModel.type` |
| `request_id` | `str` | Run endpoint's existing `request_id` |

**Behavior**: `issue()` calls
`generate_jwt_token(authorized_project_uuid, settings.JWT_SECRET_KEY)`, emits
exactly one `run_token_minted` record, and returns the token. It never logs the
token.

**Call sites**:

- Tool runs: once per test case, as today.
- Active runs: once per test case **only when** the test case's `project`
  block has no `auth_token` (FR-007). A test case that brings its own token is
  passed through unchanged and produces no audit record.
- Zero test cases: never called, so no token and no record.

## Run token (unchanged contract)

| Claim | Value |
|---|---|
| `project_uuid` | Canonical authorized project (today: `str(data.project_uuid)`, the same bytes once the check passes) |
| `exp` | `iat` + `DEFAULT_EXPIRATION_MINUTES` (2) |
| `iat` | Issue time (UTC) |

Signing stays RS256 with `JWT_SECRET_KEY`, and it is injected as
`project.auth_token` (`JWT_PROJECT_KEY`). No claim is added or removed
(FR-006). Only the docstring that says "60 minutes" changes (FR-008).

## Request rejection errors

| Type | `http_status` | `code` | Message (constant) | Raised by |
|---|---|---|---|---|
| `ProjectMismatchError` | 403 | `PROJECT_MISMATCH` | "The project in the request does not match the authorized project." | binding dependencies |
| `RunNotAttributableError` | 403 | `RUN_NOT_ATTRIBUTABLE` | "This run could not be attributed to a user." | `attributed_run_request` |

Both extend `RequestRejectedError` and carry the rejection `request_id`. They
deliberately have no `status_code` attribute, so Sentry's Starlette
integration does not capture them (research R5). One handler turns them into
the response in [contracts/http-responses.md](./contracts/http-responses.md).

## Log records

The field-level contract is in [contracts/log-events.md](./contracts/log-events.md).

| Record | Level | Cardinality |
|---|---|---|
| Mint audit record (`run_token_minted`) | INFO | Exactly one per minted token |
| Mismatch security event (`project_mismatch_rejected`) | WARNING | Exactly one per rejected mismatch |
| Unattributable run (`run_not_attributable`) | WARNING | Exactly one per run rejected for missing identity |

Invariant for every record: no run token, no bearer token, and no
`Authorization` header value (FR-014).
