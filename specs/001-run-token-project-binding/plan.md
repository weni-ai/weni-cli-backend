# Implementation Plan: Bind CLI Run Tokens to the Authorized Project

**Branch**: `001-run-token-project-binding` | **Date**: 2026-10-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-run-token-project-binding/spec.md`

## Summary

Fix security finding C1. The run endpoint authorizes the `X-Project-Uuid`
header against Connect, but mints the mesh token from the body `project_uuid`.
After this change, four endpoints (runs, agents push, channel creation and
ticketer creation) reject with HTTP 403 any request whose raw body
`project_uuid` is not byte-equal to the authorized header. Run tokens are
minted only from the canonical authorized header value. Every run must be
attributable to the `email` claim of the user's bearer token. Each minted token
leaves one structured audit line, and each rejection leaves one structured
warning. The token contract (claims, RS256, 2 minutes) and the `auth_token`
pass-through of active test cases stay exactly as they are.

Technical approach (details in [research.md](./research.md)):

- **Binding**: FastAPI dependencies that wrap each endpoint's validated body
  model and delegate to one shared check (R2). They run after
  `AuthorizationMiddleware` and before the endpoint body, keep today's 422
  priority, and compare raw strings.
- **Attribution**: a run-only dependency chained after the binding check. It
  reads `email` from the bearer JWT without verifying the signature, because
  Connect has already validated that token (R1).
- **Minting**: a `RunTokenIssuer` owns "mint plus audit". Both strategies
  receive one instead of reading `data.project_uuid` (R6).
- **Rejections**: a 403 in the CLIResponse shape with a fresh `request_id`
  (R4), raised as an exception type that Sentry does not capture (R5).
- **Logs**: an event name followed by logfmt pairs with JSON-quoted values,
  through the existing formatter (R3).

## Technical Context

**Language/Version**: Python 3.12 (CI also 3.13)

**Primary Dependencies**: FastAPI 0.115.12 / Starlette 0.46.1, Pydantic v2, PyJWT 2.12.0 (already declared), sentry-sdk 2.24.1. **No new dependency.**

**Storage**: N/A (stateless; application logs only)

**Testing**: pytest + pytest-mock + pytest-asyncio, FastAPI `TestClient`, `mock_auth_middleware` fixture; Lambda/Flows/Nexus/Gallery clients patched; RSA key generated in-test for `JWT_SECRET_KEY`

**Target Platform**: Linux container (`docker/Dockerfile`), Kubernetes namespace `cli`

**Project Type**: web-service (FastAPI backend for the Weni CLI)

**Performance Goals**: No change. Per covered request the feature adds one string comparison, one local JWT decode and at most one log line per minted token. It makes no network call.

**Constraints**: Token payload, signing and lifetime unchanged (FR-006). No change to retail-setup, agentic-cx, weni-cli or Nexus (FR-015). No token values in logs (FR-014). `jwt_generator.py`: docstring fix only.

**Scale/Scope**: Declared peak (Constitution XII): **30 requests per hour** across the four covered endpoints. This is the measured peak hour in production Loki, `cli-backend`, over the last 7 days on 2026-10-05 (research R7). The feature does not change capacity needs. Code scope: 6 new modules, 9 modified files (including `CHANGELOG.md`), and their tests.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Evidence |
|---|---|---|---|
| I | Version Control and Review | PASS | Delivered through a PR with review and green CI. Commit plan below. |
| II | Security and Secrets | PASS | No new dependency (PyJWT already locked). No new secret or setting. `JWT_SECRET_KEY` usage unchanged. |
| III | Observability | **VIOLATION (justified)** | Logs are structured (logfmt) and never contain tokens. `request_id` is in every new record. **But** `user_email` (an e-mail address) is logged in `run_token_minted` and `project_mismatch_rejected`, which conflicts with "never contain ... sensitive personal data". See Complexity Tracking. |
| IV | Versioned Contracts | PASS | `/v1` unchanged for conforming clients. The new 403s apply only to requests that conforming CLIs never send (header equals body, and real tokens carry `email`, R1). Backward compatible: SemVer PATCH. |
| V | Specification Traceability | **VIOLATION (justified)** | The spec has no "Inheritance from Product Spec" section, and no product spec for C1 exists in `vtex-cx-experience-specs`. See Complexity Tracking. |
| VI | No Silent Divergence | PASS | The plan contradicts nothing in the spec. Every plan decision fills a gap the spec delegated to planning (identity claim, log format) or was confirmed by the user. |
| VII | Commit Messages | PASS | Conventional, imperative, 50 characters or fewer, atomic (see Commit plan, amended 2026-10-05). |
| VIII | Changelog Maintenance | PASS | New `## [1.15.1]` entry with a `### Security` section. Earlier entries are not reformatted (XVI). |
| IX | Never Trust the Client | PASS | Strengthens it. The body project is validated against the server-side authorized project at the router boundary. Authorization is still enforced by `AuthorizationMiddleware`. |
| X | Fail Gracefully and Predictably | PASS | No new external call. Both rejections are explicit, consistent JSON responses with no internal details. |
| XI | Bounded Retry Over REST | N/A | No new propagation to another service. |
| XII | Scalability and Peak Load | PASS | Stateless (no state outside the request). Peak declared above. |
| XIII | Diagnosable Errors | **VIOLATION (justified)** | No new error is reported to Sentry (R5). **But** sentry-sdk's default `LoggingIntegration` attaches the `run_token_minted` line, `user_email` included, as a breadcrumb to the Sentry event of a later `logger.error` in the same run (`_yield_processing_error`). This conflicts with "e-mail addresses MUST NOT be attached to an error report". See Complexity Tracking. |
| XIV | Tests Exercise Flows | PASS | Router flow tests cover success and every failure path for all four endpoints. Unit tests cover input variations (see quickstart table). |
| XV | Explicit Over Clever | PASS | Explicit dependency per endpoint (the factory was rejected, R2). The four `bound_*` dependencies stay separate; the comment above `bound_run_request` explains why. Named constants for messages, codes and event names. Comments explain constraints: why there is no `status_code` attribute, why raw values are compared, why the JWT signature is not verified, and why those four dependencies are not collapsed. |
| XVI | Contained Changes | PASS | Feature edits are the files listed below. Existing tests change only where they encode the unsafe behavior or lack an identity (R8). `middlewares_test.py` is not touched. Commit `b18a889` also changes `app/clients/aws/logs_client.py`, which stays outside that list. |

**Pre-research gate**: two MUST violations (III, V). Both were raised with the
user and accepted as justified exceptions (session 2026-10-05). They are
recorded in Complexity Tracking, not hidden. The constitution's Governance
section still classifies an unresolved MUST conflict as CRITICAL for
`/speckit.analyze`.

**Post-design re-check**: one further MUST violation, XIII. R5 keeps both
rejections out of Sentry, but sentry-sdk's default `LoggingIntegration` attaches
the `run_token_minted` line, `user_email` included, as a breadcrumb to the
Sentry event of a later `logger.error` in the same run. Found by
`/speckit-analyze` and accepted as a justified exception (session 2026-10-05).
See Complexity Tracking.

## Design

### Request flow (covered endpoints)

```text
AuthorizationMiddleware (unchanged)
  -> FastAPI body validation (unchanged 422)
  -> bound_<endpoint>_request  ── mismatch ──> ProjectMismatchError ─┐
  -> [runs only] attributed_run_request ── no email ──> RunNotAttributableError ─┤
  -> endpoint body (unchanged behavior)                               │
                                                handle_request_rejected (403, CLIResponse)
```

### New modules

| Module | Contents |
|---|---|
| `app/api/v1/rejections.py` | `RequestRejectedError` (base: `http_status`, `code`, `message`, `request_id`; deliberately no `status_code`, R5) and `handle_request_rejected`, which returns a `JSONResponse` in the CLIResponse shape |
| `app/api/v1/user_identity.py` | `read_user_email(authorization: str \| None) -> str \| None`: `Bearer` prefix, unverified PyJWT decode, non-empty string `email`, otherwise `None` |
| `app/api/v1/project_binding.py` | `PROJECT_MISMATCH_MESSAGE`/`_CODE`, `ProjectMismatchError`, `ensure_body_project_is_authorized(request, authorized, requested_raw)` (logs `project_mismatch_rejected`, raises), and four separate dependencies: `bound_run_request` and `bound_agents_request` (Form; raw value from `request.form()`) plus `bound_channel_request` and `bound_ticketer_request` (JSON; raw value from `request.json()`). They stay separate. The comment above `bound_run_request` explains why: FastAPI validates the body from each dependency's parameter annotation, which keeps an invalid body at 422, and one shared function would drop that model. Runs and agents re-read the multipart field, and channels and ticketers re-read the JSON field, because `data.project_uuid` is already normalized. |
| `app/api/v1/run_attribution.py` | `AttributedRun` (dataclass: `request`, `authorized_project_uuid`, `user_email`), `RunNotAttributableError`, and `attributed_run_request` (depends on `bound_run_request`; logs `run_not_attributable`, raises) |
| `app/core/log_events.py` | `format_log_event(event: str, fields: Mapping[str, str \| None]) -> str`: `event=<name>` followed by `key="<json-escaped>"`, skipping `None` |
| `app/services/runs/token_issuer.py` | `RunTokenIssuer` (fields per [data-model.md](./data-model.md#run-token-issuer)). `issue()` mints with `generate_jwt_token(authorized_project_uuid, settings.JWT_SECRET_KEY)` and logs `run_token_minted` |

### Modified files

| File | Change |
|---|---|
| `app/main.py` | `app.add_exception_handler(RequestRejectedError, handle_request_rejected)` in `create_application` |
| `app/api/v1/routers/runs.py` | `data: Annotated[RunRequestModel, Form()]` becomes `run: Annotated[AttributedRun, Depends(attributed_run_request)]`. Builds a `RunTokenIssuer` from `run` and the existing `request_id`, and passes it to both strategies. Nothing else changes. |
| `app/api/v1/routers/agents.py` | `data` becomes `Annotated[ConfigureAgentsRequestModel, Depends(bound_agents_request)]` |
| `app/api/v1/routers/channels.py` | `data` becomes `Annotated[CreateChannelRequestModel, Depends(bound_channel_request)]` |
| `app/api/v1/routers/ticketers.py` | `data` becomes `Annotated[CreateTicketerRequestModel, Depends(bound_ticketer_request)]` |
| `app/services/runs/tool_strategy.py` | New `token_issuer` parameter. `generate_jwt_token(str(data.project_uuid), ...)` is replaced by `token_issuer.issue()` |
| `app/services/runs/active_strategy.py` | New `token_issuer` parameter on `run`. `build_active_test_event(test_data, token_issuer, fallback_credentials)` replaces its `project_uuid` parameter, which was used only for minting. The `JWT_PROJECT_KEY not in project` guard is unchanged (FR-007). |
| `app/services/jwt_generator.py` | Docstring: "uses default (60 minutes)" becomes "uses default (2 minutes)". Nothing else changes (FR-008). |
| `CHANGELOG.md` | `## [1.15.1]` / `### Security` entry describing the binding, the attribution requirement and the audit records |

Untouched: `AuthorizationMiddleware`, `permissions.py`, every request model,
`FlowsClient`, the configurators, `jwt_generator.py` code, and `middlewares_test.py`.

### Tests

| Location | Covers |
|---|---|
| `app/api/v1/tests/test_project_binding.py` (new) | Match, exact-string mismatch (UUID, case, formatting), event fields with and without `user_email`, value escaping, no token in logs |
| `app/api/v1/tests/test_user_identity.py` (new) | Valid email; missing header; non-Bearer scheme; non-JWT; malformed payload; missing, empty or non-string `email` |
| `app/api/v1/tests/test_run_attribution.py` (new) | Attributed result; rejection plus `run_not_attributable` record |
| `app/api/v1/tests/test_rejections.py` (new) | Handler response shape and status; the exception exposes no `status_code` (Sentry constraint) |
| `app/core/tests/test_log_events.py` (new) | Format, field order, `None` skipped, quotes/backslashes/newlines escaped |
| `app/services/runs/tests/test_token_issuer.py` (new) | The token decodes to exactly three claims with the authorized project; one record per call with the required fields; `tool_key` absent for active runs; no token in logs |
| `app/services/runs/tests/test_active_strategy.py` (updated) | `build_active_test_event` with the issuer: mints when `auth_token` is missing, passes through without minting or a record when present |
| `app/api/v1/routers/tests/test_runs.py` (updated and extended) | Fixtures send one project and a CLI-like bearer token (R8). Adds the US1 and US2 flow tests and edge cases from the quickstart table. |
| `app/api/v1/routers/tests/test_agents.py`, `test_channels.py`, `test_ticketers.py` (extended) | US3 mismatch flows (no downstream call), nested ticketer project unaffected, 422 still wins over mismatch |
| `app/api/v1/routers/tests/test_permissions.py` (extended) | `/verify` behaves as today with any body project and no header |
| `app/tests/utils.py` (extended) | `make_cli_bearer_token(email: str \| None)` (HS256 with a test-only secret of at least 32 bytes; the reader never verifies it) and `generate_rsa_key_pair()` (PEM pair built like the `rsa_key_pair` fixture in `test_jwt_generator.py`) |

### Commit plan (Constitution VII)

Amended on 2026-10-05 to follow the user-story phases of tasks.md. A baseline
`poetry run pytest -q` runs before commit 1 and produces no commit (SC-004).

1. `feat: add structured log event formatter`
2. `feat: reject body project not matching header` (rejections, user identity reader, `make_cli_bearer_token`, runs binding, handler registration, runs router, their tests and run fixture fixes)
3. `feat: require user identity for runs` (run attribution, run-router wiring, tests)
4. `feat: mint run tokens from authorized project` (issuer, both strategies, audit, `generate_rsa_key_pair`, tests)
5. `feat: bind project on push, channel, ticketer` (agents, channels and ticketers dependencies and routers, their tests)
6. `docs: fix run token lifetime in docstring`
7. `docs: add 1.15.1 security changelog entry`

## Project Structure

### Documentation (this feature)

```text
specs/001-run-token-project-binding/
├── spec.md
├── plan.md              # This file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   ├── http-responses.md
│   ├── log-events.md
│   └── run-token.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 (/speckit-tasks, not created here)
```

### Source Code (repository root)

```text
app/
├── main.py                              # modified: register rejection handler
├── core/
│   ├── log_events.py                    # new
│   └── tests/test_log_events.py         # new
├── api/v1/
│   ├── rejections.py                    # new
│   ├── user_identity.py                 # new
│   ├── project_binding.py               # new
│   ├── run_attribution.py               # new
│   ├── tests/
│   │   ├── test_rejections.py           # new
│   │   ├── test_user_identity.py        # new
│   │   ├── test_project_binding.py      # new
│   │   └── test_run_attribution.py      # new
│   └── routers/
│       ├── runs.py agents.py channels.py ticketers.py   # modified
│       └── tests/test_runs.py test_agents.py test_channels.py test_ticketers.py test_permissions.py  # modified
├── services/
│   ├── jwt_generator.py                 # docstring only
│   └── runs/
│       ├── token_issuer.py              # new
│       ├── tool_strategy.py active_strategy.py          # modified
│       └── tests/test_token_issuer.py test_active_strategy.py
└── tests/utils.py                       # modified: bearer-token and RSA key-pair helpers
CHANGELOG.md                             # modified
```

Commit `b18a889` also changes `app/clients/aws/logs_client.py` (a mypy `cast` on `get_query_results`). That file is outside the feature file list above. It is not part of this feature's scope, and it is not added to the list. The commit stays as it is.

**Structure Decision**: Keep the existing single-service layout (Constitution,
Technology and Layout). Request-boundary concerns (binding, attribution,
identity reading, rejection responses) live in `app/api/v1`. Run-token issuing
is a run use case and lives in `app/services/runs`. The log-line formatter is
shared, so it lives in `app/core`. Tests sit beside the code, as in the
existing suite.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| **V**: no product spec and no inheritance section | C1 is a finding of an internal security review, with no product intent to inherit. The user decided to treat it as a justified exception (session 2026-10-05). | Writing and tagging a product spec in `vtex-cx-experience-specs` first was offered and declined by the user. Under Governance, `/speckit.analyze` will still report this as CRITICAL until the spec gains an inheritance section or the constitution is amended. |
| **III**: `user_email` (personal data) in `run_token_minted` and `project_mismatch_rejected` | FR-010/FR-012/SC-003 need a user identity that is present in every real CLI token. `email` is the platform's user key (Connect, Flows) and production behavior proves it is present (R1). The user chose it explicitly. | `sub` is opaque and compliant, but its presence in `weni-cli` tokens is not proven. Under FR-011 a missing claim would make every run fail. Its exposure through Sentry breadcrumbs is tracked in the XIII row. |
| **XIII**: `user_email` reaches Sentry as a breadcrumb of `run_token_minted` when a run fails after minting | Same need as III (FR-010/FR-012/SC-003). The breadcrumb comes from sentry-sdk's default `LoggingIntegration`. The user chose to accept it (session 2026-10-05). | Keeping the audit lines out of Sentry breadcrumbs was offered and declined by the user (session 2026-10-05). Under Governance, `/speckit.analyze` will still report this as CRITICAL until the constitution is amended. |
