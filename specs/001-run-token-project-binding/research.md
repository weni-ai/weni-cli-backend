# Research: Bind CLI Run Tokens to the Authorized Project

**Feature**: `001-run-token-project-binding` | **Date**: 2026-10-05 | **Plan**: [plan.md](./plan.md)

Every item below was resolved from the code of this repository, the sibling
repositories checked out next to it (`weni-cli`, `weni-engine`, `flows`,
`connect-keycloak`, `vtex-cx-experience-specs`), production logs in Grafana
Loki, throwaway probes run against the locked FastAPI version, or an explicit
decision from the user during the planning session. Nothing is assumed.

Locked versions that the findings depend on (`poetry.lock`): FastAPI 0.115.12,
Starlette 0.46.1, PyJWT 2.12.0, sentry-sdk 2.24.1, python-multipart 0.0.20.

---

## R1. Which bearer-token claim is the user identity

**Decision**: The `email` claim of the user's Keycloak access token, read
locally with `jwt.decode(token, options={"verify_signature": False})` from
PyJWT, which is already a dependency. No network call.

**Rationale**:

- The CLI logs in through the Keycloak authorization-code flow (realm `weni`,
  client `weni-cli`, Keycloak 26.7.3, no explicit `scope` parameter) and sends
  the resulting access token as `Authorization: Bearer <token>`
  (`weni-cli/weni_cli/auth.py`, `weni_cli/clients/cli_client.py`).
- `AuthorizationMiddleware` sends that same header to Connect
  (`/v2/projects/{uuid}/authorization`). Connect authenticates it against
  Keycloak userinfo (`mozilla-django-oidc`), and identifies users by `email`
  (`weni-engine/connect/middleware.py`). Connect itself decodes the same access
  token without signature verification to read a claim. A request that reaches
  a router has therefore already been validated by Connect, so a local
  unverified decode reads a validated token.
- **Presence in real CLI login tokens is confirmed by production behavior**:
  `FlowsClient._extract_email_from_token` reads `email` from the CLI access
  token and sends it to Flows as `user`. Flows' ticketer endpoint
  (`flows/temba/api/v2/internals/tickets/views.py`) looks the user up with
  `User.objects.get(email=user_email)`, returns 400 `User not found`
  otherwise, and also rejects a payload user that differs from the
  authenticated user. Loki shows 52 successful channel/ticketer creations by
  `cli-backend` in the last 30 days, with zero `Failed to extract email` warnings
  and zero `Error creating channel/ticketer` lines. Real CLI tokens carry a
  correct `email`.
- The user chose `email` explicitly (planning session, 2026-10-05) and accepted
  the conflict with Constitution III/XIII recorded in the plan's Complexity
  Tracking.

**Resolution rule (FR-010/FR-011)**: The identity is resolved when the
`Authorization` header is `Bearer <jwt>`, the payload decodes, and `email` is a
non-empty string. Any other case (missing header, other scheme, malformed JWT,
missing/empty/non-string `email`) means "identity cannot be resolved".

**Alternatives considered**:

- `sub` (opaque Keycloak id): it complies with Constitution III/XIII, but no
  code or production evidence confirms that it appears in `weni-cli` tokens.
  The user rejected it in favor of `email`.
- `preferred_username`: Connect uses it only as the username fallback. Its
  presence in CLI tokens is not proven and it is not the platform's user key.
- Calling Keycloak userinfo or Connect: forbidden by FR-010 (no extra call).
- Reusing `FlowsClient._extract_email_from_token`: it is a private method of a
  client and returns `""` on failure. Moving or refactoring it is outside this
  feature (Constitution XVI). The new reader uses PyJWT, the same way Connect
  does.

---

## R2. Where the header/body project check belongs

**Decision**: A FastAPI dependency per covered endpoint, all delegating to one
shared check, `ensure_body_project_is_authorized`, in
`app/api/v1/project_binding.py`. Each endpoint replaces its body parameter with
`Annotated[<Model>, Depends(<bound_dependency>)]`. The dependency declares the
same validated body model (`Form()` for runs and agents, JSON for channels and
ticketers) plus the `X-Project-Uuid` header. It compares the header with the
**raw** body string and raises `ProjectMismatchError` on any difference.

**Rationale** (verified with throwaway probes against the locked FastAPI):

- **Runs after authorization**: `AuthorizationMiddleware` wraps the whole app,
  so every dependency runs only for requests that passed Connect.
- **Runs before any processing (FR-002)**: dependencies are solved before the
  endpoint body. The endpoint body is the only place where forms are read,
  Lambda clients are built, tools are packaged, or Flows/Nexus/Gallery are
  called. Probe: on a mismatch the endpoint was never entered.
- **Validation errors keep priority (edge case "body project missing or not a
  valid UUID")**: when the body model fails validation, FastAPI skips the
  dependency and returns today's 422. Probe: a missing or invalid body UUID
  returned 422 and the dependency was not called, for both Form and JSON.
- **Exact string comparison (clarification)**: Pydantic's `UUID4` normalizes
  input, so comparing against `str(data.project_uuid)` let an upper-case body
  pass (probe: 200). The check therefore reads the raw value. For Form bodies
  it uses `(await request.form())["project_uuid"]` (Starlette caches the parsed
  form on the request). For JSON bodies it uses
  `(await request.json())["project_uuid"]` (the body bytes are cached). Probe:
  case-only differences return 403 for both. Uploaded files remain readable by
  the endpoint afterwards.
- **Middleware is not used**: a middleware would have to consume the request
  stream before FastAPI (multipart and JSON) and re-inject it, and it would run
  before FastAPI's body validation. That would break the 422 priority rule
  above.

**Run identity ordering**: `attributed_run_request` (in
`app/api/v1/run_attribution.py`) is the run endpoint's single dependency. It
depends on `bound_run_request`, so FastAPI runs the binding check first and
skips identity resolution when the body is invalid. Probe results: a mismatch
together with an unreadable identity returns 403 `PROJECT_MISMATCH`, and the
mismatch event is logged without an identity. An invalid body together with an
unreadable identity returns 422. A matching project with an unreadable identity
returns 403 `RUN_NOT_ATTRIBUTABLE` before the endpoint runs, so no form is read,
no Lambda is created and no token is minted (FR-011, including active runs and
runs with zero test cases).

**Alternatives considered**:

- One line `ensure_...(...)` call at the top of each endpoint body. It is a
  shared function too, but it runs after `request.form()` is in scope and
  relies on every future endpoint remembering to call it first. The dependency
  makes the binding part of the endpoint signature.
- A generic dependency factory that builds `Annotated[model, Form()]`
  dynamically. It removes four short functions but hides the body source behind
  runtime-built annotations (Constitution XV, Explicit over Clever).
- Comparing parsed UUIDs: rejected because it contradicts the exact-string
  clarification.

---

## R3. Structured log format for the mint audit record and the mismatch event

**Decision**: Each record is one log line whose message is a fixed event name
followed by logfmt `key="value"` pairs. Every value is rendered as a JSON
string literal (quotes, backslashes and control characters escaped). A field
whose value is unknown is omitted. Records are emitted through the module
`logger` with the existing root formatter (`%(asctime)s %(levelname)-8s
%(message)s`). A small shared helper, `format_log_event(event, fields)` in
`app/core/log_events.py`, renders the message.

**Rationale**:

- Production lines in Loki (`{service_name="cli-backend"}`) are plain text in
  exactly that format, with the level detected by Loki. No JSON formatter is
  configured anywhere in the service, and the user chose not to introduce one
  (it would change every log line, outside this feature's scope).
- LogQL's `| logfmt` parser already extracts `key=value` pairs from those
  existing lines (verified on the `apm_instrumentation=` lines of the agents
  router). Operators can therefore filter and aggregate by `event`, `user_email`,
  `request_id` and the other fields with no pipeline change (SC-003, SC-006).
- Agent and tool keys and the raw body project come from the client.
  Quoting with JSON escaping blocks log-field injection, for example an agent
  key containing ` user_email="victim"` or a newline. Loki's logfmt decoder
  (go-logfmt) accepts the same escapes as JSON strings.
- Token values are never passed to the helper. Tests assert that no minted
  token or bearer string appears in captured log output (FR-014, SC-005).

**Events** (field-level contract in [contracts/log-events.md](./contracts/log-events.md)):

| Event name | Level | When |
|---|---|---|
| `run_token_minted` | INFO | Once per minted run token |
| `project_mismatch_rejected` | WARNING | Once per rejected header/body mismatch |
| `run_not_attributable` | WARNING | Once per run rejected because identity cannot be resolved |

`run_not_attributable` follows from Constitution III: errors must be traceable
by correlation id. The user chose not to report these rejections to Sentry (R5),
so this log line is the only trace of the failure. It carries no identity,
because none could be read, and no token.

**Alternatives considered**: a JSON object as the message (Loki's `| json`
cannot parse it behind the plain-text prefix). A global JSON formatter (outside
the feature's scope). `extra=` fields only (the current formatter drops them,
so they would never reach Loki).

---

## R4. Response body for the two new rejections

**Decision**: The response body uses the CLIResponse shape: `message`, `data:
null`, `success: false`, `code`, `request_id`. Each rejection gets a fresh
`request_id` (uuid4) generated at rejection time, and the same id appears in
the matching log event.

**Rationale**: For runs and agents push the CLI displays `message` (falling back
to `detail`). For channels and ticketers it displays only `message`, and it
replaces any 401 with its own "please login again" text
(`weni-cli/weni_cli/clients/cli_client.py`). A `message` key with status 403
makes the spec's text visible on all four commands. Channels and ticketers have
no request id today, so the rejection mints one, which satisfies FR-013's
request id for every covered endpoint. The user chose this shape and HTTP 403
for both rejections (planning session).

**Alternatives considered**: FastAPI's default `{"detail": ...}` (channels and
ticketers would show a generic "status 403"). HTTP 401 for the identity
failure (the CLI would hide the spec's message). A streamed error event with
HTTP 200 (the rejection happens before streaming starts, so a plain response is
simpler and matches the mismatch response).

---

## R5. Sentry and the new rejections

**Decision**: Neither rejection is reported to Sentry. Both are raised as
`RequestRejectedError` subclasses whose status is stored in `http_status`, not
`status_code`, and handled by one exception handler registered in
`create_application`.

**Rationale**: sentry-sdk 2.24.1 captures a handled exception only when it has
an integer `status_code` in `failed_request_status_codes`
(`integrations/starlette.py`, `_sentry_patched_exception_handler`). The app
sets that range to 401–598 and `send_default_pii=True`, which sends request
headers unfiltered (`_filter_headers`). An `HTTPException(403)` would therefore
ship the user's bearer token and e-mail to Sentry on every rejection. The
structured logs are the record of these events. The user chose this option. The
Sentry settings themselves (the `TODO(SENTRY_PII)` in the constitution) remain
out of scope.

**Limit**: this covers exception capture only. sentry-sdk's default
`LoggingIntegration` records INFO and higher lines as breadcrumbs and sends
ERROR lines as events, so a `run_token_minted` line (with `user_email`) is
attached to the Sentry event of a later `logger.error` in the same run.
Accepted as a Constitution XIII exception (plan.md, Complexity Tracking).

**Alternative considered**: `HTTPException(403)` like other 4xx responses.
Rejected by the user because of the header exposure.

---

## R6. Project value placed in the minted token

**Decision**: `str(UUID(x_project_uuid))`, the canonical form of the
authorized header value.

**Rationale**: FR-004 requires the authorized header as the only source.
`weni project use` stores the UUID exactly as typed, and Connect's route takes
it as a plain string, so the header can legitimately be upper case. Today's
token carries `str(data.project_uuid)`, which is lower-case and hyphenated.
After the binding check the header is byte-equal to the raw body, which already
passed `UUID4` validation, so canonicalizing it cannot fail. It also yields the
exact value tokens carry today (FR-006, SC-004). The user chose this option.

**Alternative considered**: the raw header string. Rejected because an
upper-case header would change the claim value that retail-setup receives.

---

## R7. Peak load (Constitution XII)

**Decision**: Declare the measured peak and state that the feature does not
change capacity needs.

**Evidence**: Loki, `cli-backend`, last 7 days, hourly counts of the four
covered operations (`Processing test run`, `Processing agent configuration`,
`Creating channel`, `Creating ticketer`): the peak was **30 requests per hour**.
Per request the feature adds one string comparison, one local JWT decode, and
one log line per minted token. It keeps no state, so the service stays
stateless.

---

## R8. Test strategy and impact on existing tests

**Decision**: Follow the existing patterns. Router flow tests use `TestClient`
with the `mock_auth_middleware` fixture, and patch `AWSLambdaClient` and
`process_tool`/`ActiveAgentProcessor` to stay off AWS. Flows, Nexus and Gallery
clients are patched to stay off HTTP. Unit tests sit beside each new module.
A test helper builds CLI-like bearer tokens with PyJWT (HS256 with a test-only
secret, which the reader never verifies). Flow tests that decode
the injected run token patch `settings.JWT_SECRET_KEY` with a generated RSA
key pair, as `app/services/tests/test_jwt_generator.py` already does.

**Existing tests that must change** (they encode today's unsafe behavior or
lack an identity):

- `app/api/v1/routers/tests/test_runs.py`: the module-scoped `auth_header`
  fixture sets `X-Project-Uuid` to `str(project_uuid)`, where `project_uuid` is
  the fixture function itself, while the body uses a random `uuid4()`. Several
  tests also send a fresh `uuid4()` header. All of these become mismatches.
  `TEST_TOKEN = "Bearer test-token"` is not a JWT, so every run would be
  `RUN_NOT_ATTRIBUTABLE`. The fixtures must send one project in both places and
  a CLI-like bearer token. Tests that patch
  `app.services.runs.tool_strategy.generate_jwt_token` move to the new issuer
  module.
- `app/api/v1/routers/tests/test_agents.py`: data and header already share
  `TEST_PROJECT_UUID`. Verify, and change only where they differ.
- `app/api/v1/routers/tests/test_channels.py` and `test_ticketers.py`: the
  invalid-body tests send a different header UUID, and they must still return
  422 (validation runs first). The other tests already match.
- `app/services/runs/tests/test_active_strategy.py`: `build_active_test_event`
  takes the token issuer instead of `project_uuid`, so its patches of
  `active_strategy.generate_jwt_token` change accordingly.

**Observation, not changed**: `app/api/v1/middlewares_test.py` is collected
today. pytest 8.3.5 reads only `[tool.pytest.ini_options]`, so the
`python_files = ["test_*.py"]` under `[tool.pytest]` is ignored and the default
`*_test.py` pattern applies. It holds the role-check tests that FR-009 relies
on. This feature does not modify `AuthorizationMiddleware`, so that file stays
as it is (Constitution XVI).
