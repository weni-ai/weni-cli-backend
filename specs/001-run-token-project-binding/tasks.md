---

description: "Task list for Bind CLI Run Tokens to the Authorized Project"
---

# Tasks: Bind CLI Run Tokens to the Authorized Project

**Input**: Design documents from `specs/001-run-token-project-binding/`

**Prerequisites**: plan.md, spec.md, research.md (R1–R8), data-model.md, contracts/ (http-responses.md, log-events.md, run-token.md), quickstart.md, `.specify/memory/constitution.md`

**Tests**: REQUIRED. Constitution XIV requires a flow test for every flow (success and failure paths) under `app/api/v1/routers/tests`, plus unit tests beside each new module. Every row of the quickstart scenario table maps to at least one test task (see "Quickstart scenario coverage" below). Tests stay off live HTTP and AWS: use the `mock_auth_middleware` fixture, patch `app.api.v1.routers.runs.AWSLambdaClient`, `process_tool`, `ActiveAgentProcessor`, `FlowsClient`, `ConnectClient` and the agent configurators, and patch `settings.JWT_SECRET_KEY` with a generated RSA key when a test decodes a minted token. Log assertions use pytest's built-in `caplog` fixture with `caplog.set_level(logging.INFO)`. Commands use `poetry run`, which needs Poetry 1.8 or later because `pyproject.toml` sets `package-mode` (CI pins 1.8.5). With an older Poetry, run the same commands through the existing virtualenv's `python -m` (quickstart.md, Prerequisites).

**Organization**: Tasks are grouped by user story. Each phase is annotated with the commit it belongs to.

**Accepted, not to be "fixed"**: the Constitution III violation (`user_email` in log records), the Constitution V violation (no product spec) and the Constitution XIII violation (`user_email` in Sentry breadcrumbs) are recorded in plan.md Complexity Tracking. No task addresses them.

**Do not touch** (Constitution XVI): `AuthorizationMiddleware` (`app/api/v1/middlewares.py`), `app/api/v1/routers/permissions.py`, `app/api/v1/models/requests.py`, `app/clients/flows_client.py`, the agent configurators, `app/api/v1/middlewares_test.py`, `app/services/tests/test_jwt_generator.py`, and the code of `app/services/jwt_generator.py` (docstring only). No new dependency.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other [P] tasks of its phase once its listed dependencies are done (different files)
- **[Story]**: Which user story the task belongs to (US1, US2, US3)

## Commit plan (reconciled with the user-story phases)

plan.md originally defined six commits; its Commit plan now matches the table below. Organizing by user story required four changes, all decided by the user on 2026-10-05: original plan commit 3 is split and its US3 part becomes a new commit after original plan commit 5; `user_identity.py`, its test and `make_cli_bearer_token` move from original plan commit 4 into original plan commit 3, because the mismatch event logs `user_email` when it can be read (FR-013); the docstring fix (original plan commit 1) stays in Polish and becomes the sixth commit; and `generate_rsa_key_pair()` is added to `app/tests/utils.py` in original plan commit 5. A baseline test run (T001) precedes commit 1.

| Order | Commit message | Original plan.md ref | Tasks |
|---|---|---|---|
| 1 | `feat: add structured log event formatter` | #2 | T002–T003 |
| 2 | `feat: reject body project not matching header` | #3 (runs only, plus user identity) | T004–T015 |
| 3 | `feat: require user identity for runs` | #4 (without user identity) | T016–T020 |
| 4 | `feat: mint run tokens from authorized project` | #5 (plus `generate_rsa_key_pair`) | T021–T030 |
| 5 | `feat: bind project on push, channel, ticketer` | new (split from #3) | T031–T039 |
| 6 | `docs: fix run token lifetime in docstring` | #1 | T040 |
| 7 | `docs: add 1.15.1 security changelog entry` | #6 | T041 |

T001 (baseline) and T042 (quality gate) produce no commit.

---

## Phase 1: Setup

**Purpose**: Establish a green baseline so later failures are attributable to this feature (SC-004).

**Commit**: none.

- [ ] T001 Run `poetry run pytest -q` from the repository root on branch `001-run-token-project-binding` before any code change, and confirm the suite is green. If anything fails, stop and report it before continuing; it is not caused by this feature.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared building blocks used by every story: the log line formatter, the rejection exception and its 403 handler, the bearer-token identity reader, and the test helper that builds CLI-like bearer tokens.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

### Commit 1: `feat: add structured log event formatter` (original plan #2)

- [ ] T002 [P] Write unit tests for `format_log_event` in `app/core/tests/test_log_events.py`. Cover: the message starts with `event=<name>`; fields follow the mapping's insertion order; a field whose value is `None` is omitted entirely (no empty or placeholder value); every value is rendered as a JSON string literal, so `"`, `\`, `\n`, `\r` and other control characters are escaped and the result stays on one line; a value such as `x" user_email="victim` stays inside one quoted field and cannot add a field; an empty mapping yields just `event=<name>`. Satisfies research R3 and the "Line format" rules in contracts/log-events.md, and blocks field and line injection into audit lines (contracts/log-events.md, Line format).
- [ ] T003 [P] Create `app/core/log_events.py` with `format_log_event(event: str, fields: Mapping[str, str | None]) -> str`. It returns `event=<event>` followed by one space-separated `key=<json.dumps(value)>` pair per field, in mapping order, skipping fields whose value is `None`. No logger in this module; callers log the returned string through their own module `logger`, so the existing root formatter (`%(asctime)s %(levelname)-8s %(message)s`, `app/core/config.py`) is used unchanged (R3).

Commit T002–T003 with `feat: add structured log event formatter` once `poetry run pytest -q`, `poetry run ruff check .` and `poetry run mypy .` are green.

### Part of commit 2: `feat: reject body project not matching header` (original plan #3, completed at the end of Phase 3)

- [ ] T004 [P] Write unit tests in `app/api/v1/tests/test_rejections.py`. (1) Construct a `RequestRejectedError(http_status=403, code="SOME_CODE", message="Some message.", request_id=<uuid4 str>)`, call `await handle_request_rejected(request, error)` with a Starlette `Request` built from a minimal HTTP scope, and assert: status code 403, `application/json` content type, and a JSON body exactly equal to `{"message": ..., "data": None, "success": False, "code": ..., "request_id": ...}` with no other keys (no `timestamp`). (2) Assert `not hasattr(error, "status_code")`, which is the Sentry constraint from R5. (3) Assert `app.main.app.exception_handlers[RequestRejectedError] is handle_request_rejected`, which proves the registration in `create_application`. Satisfies research R4/R5 and the response shape in contracts/http-responses.md.
- [ ] T005 [P] Create `app/api/v1/rejections.py`. Define `RequestRejectedError(Exception)` with `__init__(self, http_status: int, code: str, message: str, request_id: str)`, storing the four values as attributes and passing `message` to `Exception`. Add one comment explaining why the status lives in `http_status`: sentry-sdk 2.24.1's Starlette integration captures handled exceptions that carry an integer `status_code` within `failed_request_status_codes` (401–598 in `app/main.py`), and with `send_default_pii=True` that would ship the bearer token and e-mail to Sentry (R5). Define `async def handle_request_rejected(request: Request, exc: Exception) -> JSONResponse`. It builds a `CLIResponse` (`app/core/response.py`) with `message`, `data=None`, `success=False`, `code` and `request_id` taken from the error, and returns `JSONResponse(status_code=exc.http_status, content=body)`. Type the signature so `app.add_exception_handler` passes the repository's strict mypy.
- [ ] T006 Register the handler in `create_application` in `app/main.py`: `app.add_exception_handler(RequestRejectedError, handle_request_rejected)`, importing both from `app.api.v1.rejections`. Change nothing else in the file; the Sentry initialization stays as it is (out of scope). Depends on T005.
- [ ] T007 [P] Add `make_cli_bearer_token(email: str | None) -> str` to `app/tests/utils.py`. It returns the full `Authorization` header value, `"Bearer " + jwt.encode(payload, <test secret>, algorithm="HS256")`, where `payload` is `{"email": email}` when `email` is not `None` and `{}` otherwise. Put the secret in a named module constant, marked as test-only and at least 32 bytes long (PyJWT 2.12 emits `InsecureKeyLengthWarning` for shorter HMAC keys). The return value replaces `TEST_TOKEN = "Bearer test-token"` in run tests (research R8). PyJWT is already a dependency.
- [ ] T008 Write unit tests for `read_user_email` in `app/api/v1/tests/test_user_identity.py`, using `make_cli_bearer_token` from T007 for valid tokens and `jwt.encode` directly for payloads the helper can't express. Cases: a valid `Bearer <jwt>` with `email` returns the e-mail. All of the following return `None` without raising: `None` header; empty string; a non-Bearer scheme (`Basic abc`); `Bearer` with no token; `Bearer not-a-jwt`; a three-segment token whose payload segment isn't base64 JSON; a token whose payload is valid JSON but not an object; a token without `email`; `email` equal to `""`; `email` that is not a string (for example `123` or a list). Satisfies FR-010 and the R1 resolution rule. Depends on T007.
- [ ] T009 [P] Create `app/api/v1/user_identity.py` with a `BEARER_PREFIX = "Bearer "` constant and `read_user_email(authorization: str | None) -> str | None`. Return `None` unless the header starts with `BEARER_PREFIX`. Decode the remainder with `jwt.decode(token, options={"verify_signature": False})`, catching `jwt.PyJWTError` and returning `None`. Return `email` only when it is a non-empty `str`. Add one comment explaining why the signature isn't verified: `AuthorizationMiddleware` has already had Connect validate this exact token, so the local decode reads a validated token without an extra call (R1, FR-010). It makes no network call and never logs.

**Checkpoint**: Formatter committed. Rejection handler, identity reader and bearer helper are ready (staged for commit 2).

---

## Phase 3: User Story 1 – A user cannot obtain a token for a project they are not authorized in (Priority: P1) 🎯 MVP

**Goal**: The run endpoint (tool and active) rejects with 403 `PROJECT_MISMATCH` any request whose raw body `project_uuid` isn't byte-equal to `X-Project-Uuid`, before any form processing, Lambda creation or minting. It logs one `project_mismatch_rejected` warning.

**Independent Test**: With header A and body B, a tool run and an active run each get 403 `PROJECT_MISMATCH`, no Lambda client is built, `process_tool`/`ActiveAgentProcessor` are never called, nothing is minted, and exactly one mismatch event is logged per attempt.

**Commit**: completes commit 2, `feat: reject body project not matching header`, together with T004–T009.

### Tests for User Story 1

- [ ] T010 [P] [US1] Fix the run test fixtures in `app/api/v1/routers/tests/test_runs.py` (research R8), mirroring `test_agents.py`. Add a module constant `TEST_PROJECT_UUID` (a fixed, lower-case canonical UUID4 literal, so `str(UUID(TEST_PROJECT_UUID)) == TEST_PROJECT_UUID`) and `TEST_USER_EMAIL = "dev@example.com"`. Make the `project_uuid` fixture module-scoped, returning `UUID(TEST_PROJECT_UUID)`. Make `auth_header` take `project_uuid` (today it stringifies the fixture function itself) and send `"Authorization": make_cli_bearer_token(TEST_USER_EMAIL)`. Use `str(project_uuid)` in `run_tool_request_data` and `run_active_request_data` instead of `str(uuid4())`. In `test_validation_errors`, the `missing_tool_file` case uses `TEST_PROJECT_UUID`. `test_agent_not_found_in_definition` and `test_tool_not_found_for_agent` use the `auth_header` fixture instead of inline `TEST_TOKEN`/`str(uuid4())` headers. Remove `TEST_TOKEN` once nothing uses it. Change no assertion: these tests must keep passing unchanged in behavior after binding lands (SC-004).
- [ ] T011 [P] [US1] Write unit tests for `ensure_body_project_is_authorized` in `app/api/v1/tests/test_project_binding.py`, building Starlette `Request` objects from a minimal HTTP scope (path `/api/v1/runs`, `Authorization` header from `make_cli_bearer_token`). Cases: (a) identical strings return `None` with no log record. (b) Mismatches raise `ProjectMismatchError`: a different UUID, the same UUID in upper case, and the same UUID without hyphens. (c) The raised error has `http_status == 403`, `code == PROJECT_MISMATCH_CODE`, `message == PROJECT_MISMATCH_MESSAGE`, a `request_id` that parses as a UUID, and no `status_code` attribute (R5). Neither project value appears in the message (FR-003). (d) Exactly one WARNING record starts with `event=project_mismatch_rejected`, with fields in contract order: `header_project_uuid`, `body_project_uuid`, `endpoint` (the request path), `request_id` (equal to the error's), `user_email`. (e) `user_email` is absent when the token has no `email`, when the header isn't a JWT, and when there is no `Authorization` header; the error is raised in every case. (f) A raw body value containing `"` and a newline is escaped and stays on one line. (g) The bearer token string never appears in `caplog.text` (FR-014). Satisfies FR-001, FR-003, FR-013, FR-014 and US1 scenarios 3–4. Depends on T007.
- [ ] T012 [US1] Add a `TestRunProjectBinding` class with flow tests to `app/api/v1/routers/tests/test_runs.py`, using `mock_auth_middleware` and patching `app.api.v1.routers.runs.AWSLambdaClient`, `app.services.runs.tool_strategy.process_tool` (repository `AsyncMock`), `app.services.runs.active_strategy.ActiveAgentProcessor`, and both `tool_strategy.generate_jwt_token` and `active_strategy.generate_jwt_token`. Tests: (1) A tool run with header `TEST_PROJECT_UUID` and a different body UUID returns 403 and a JSON body exactly `{"message": PROJECT_MISMATCH_MESSAGE, "data": None, "success": False, "code": "PROJECT_MISMATCH", "request_id": <uuid>}`. Neither UUID appears in `response.text`. `AWSLambdaClient` is never instantiated, `process_tool.call_count == 0`, the mint is never called, and exactly one `event=project_mismatch_rejected` WARNING record exists (US1-1, FR-001–FR-003, SC-001). (2) The same for an active run, with `ActiveAgentProcessor` never called (US1-2). (3) A body equal to `TEST_PROJECT_UUID.upper()` returns 403 `PROJECT_MISMATCH` (US1-3). (4) The event has `header_project_uuid`, `body_project_uuid`, `endpoint == api_path`, `request_id` equal to the response's, and `user_email == TEST_USER_EMAIL`. With `make_cli_bearer_token(None)` the response is still 403 `PROJECT_MISMATCH` and the event has no `user_email` field (US1-4, FR-013). (5) A body `project_uuid` of `not-a-uuid`, and a body without `project_uuid`, each with header `TEST_PROJECT_UUID`, return 422 with no `project_mismatch_rejected` record (edge case "body project missing or not a valid UUID"). (6) In every test of this class, the bearer token string doesn't appear in `caplog.text` (FR-014, SC-005). (7) A tool run with header A, body B and no `tool` file, and an active run with header A, body B and no resources, each return 403 `PROJECT_MISMATCH` instead of today's 400 (edge case "mismatch combined with other invalid input"). Depends on T010.

### Implementation for User Story 1

- [ ] T013 [P] [US1] Create `app/api/v1/project_binding.py`. Define named constants (Constitution XV): `PROJECT_MISMATCH_MESSAGE = "The project in the request does not match the authorized project."`, `PROJECT_MISMATCH_CODE = "PROJECT_MISMATCH"`, and the event name `project_mismatch_rejected`. Define `ProjectMismatchError(RequestRejectedError)` with `__init__(self, request_id: str)`, passing `http_status=status.HTTP_403_FORBIDDEN`, the code and the message. Define `ensure_body_project_is_authorized(request: Request, authorized_project_uuid: str, requested_project_uuid_raw: str) -> None`. It returns when the two strings are equal. Otherwise it generates `request_id = str(uuid4())`, logs `logger.warning(format_log_event(<event>, {"header_project_uuid": ..., "body_project_uuid": ..., "endpoint": request.url.path, "request_id": request_id, "user_email": read_user_email(request.headers.get("Authorization"))}))`, and raises `ProjectMismatchError(request_id)`. Add one comment explaining why raw strings are compared: `UUID4` validation normalizes letter case, and the clarification requires exact string comparison (R2). Define `async def bound_run_request(request: Request, data: Annotated[RunRequestModel, Form()], x_project_uuid: Annotated[str, Header()]) -> RunRequestModel`. It reads the raw value with `(await request.form())["project_uuid"]` (Starlette caches the parsed form, so the endpoint still reads it and its files afterwards), calls `ensure_body_project_is_authorized(request, x_project_uuid, raw)` and returns `data`. Narrow the form value's type so mypy passes. Because FastAPI validates `data` before running the dependency, today's 422 keeps priority (R2). Satisfies FR-001, FR-002, FR-003, FR-013. Depends on T003, T005, T009.
- [ ] T014 [US1] In `run_test` in `app/api/v1/routers/runs.py`, replace `data: Annotated[RunRequestModel, Form()]` with `data: Annotated[RunRequestModel, Depends(bound_run_request)]`. Import `Depends` and `bound_run_request`, and drop the `Form` import if it becomes unused. The endpoint body doesn't change. This intermediate signature is replaced by `attributed_run_request` in T018. Satisfies FR-001/FR-002 for tool and active runs. Depends on T013.
- [ ] T015 [US1] Run `poetry run pytest -q`, `poetry run ruff check .` and `poetry run mypy .`. All existing run, agents, channels, ticketers and permissions tests must stay green, alongside T004, T008, T011 and T012. Then commit T004–T014 with `feat: reject body project not matching header`.

**Checkpoint**: C1 is closed for runs. A cross-project run can no longer mint a token for another project.

---

## Phase 4: User Story 2 – Legitimate runs keep working unchanged (Priority: P1)

**Goal**: Every run is attributable to the bearer token's `email`, or it fails with 403 `RUN_NOT_ATTRIBUTABLE` before anything is built. Tokens are minted only by `RunTokenIssuer` from the canonical authorized header, with an unchanged contract and one `run_token_minted` audit line per token. Active test cases with their own `auth_token` pass through untouched.

**Independent Test**: A tool run with N test cases and an active run, each with matching projects and a CLI-like bearer token, stream as before. Each injected token decodes to exactly `project_uuid`/`exp`/`iat` with the authorized project, and there is one audit record per minted token.

**Depends on**: US1 (`attributed_run_request` depends on `bound_run_request`; T010 fixtures).

### Commit 3: `feat: require user identity for runs` (original plan #4)

- [ ] T016 [P] [US2] Write unit tests for `attributed_run_request` in `app/api/v1/tests/test_run_attribution.py`. Call it directly (`await attributed_run_request(request, data, x_project_uuid)`) with a valid `RunRequestModel` and a Starlette `Request` whose headers carry `Authorization`. Cases: (a) with `make_cli_bearer_token("dev@example.com")` it returns `AttributedRun(request=data, authorized_project_uuid=str(UUID(header)), user_email="dev@example.com")`. (b) An upper-case header yields a lower-case canonical `authorized_project_uuid` (R6). (c) With `make_cli_bearer_token(None)` it raises `RunNotAttributableError` with `http_status == 403`, `code == RUN_NOT_ATTRIBUTABLE_CODE`, `message == RUN_NOT_ATTRIBUTABLE_MESSAGE` and a UUID `request_id`, and the error has no `status_code` attribute (R5). (d) That rejection writes exactly one WARNING `event=run_not_attributable` record with `header_project_uuid` (the raw header), `endpoint` (the request path) and `request_id` equal to the error's, and no `user_email` field. (e) The bearer string never appears in `caplog.text`. Satisfies FR-010, FR-011, FR-014.
- [ ] T017 [P] [US2] Create `app/api/v1/run_attribution.py`. Define named constants `RUN_NOT_ATTRIBUTABLE_MESSAGE = "This run could not be attributed to a user."`, `RUN_NOT_ATTRIBUTABLE_CODE = "RUN_NOT_ATTRIBUTABLE"` and the event name `run_not_attributable`. Define the dataclass `AttributedRun` with fields `request: RunRequestModel`, `authorized_project_uuid: str` and `user_email: str`. Define `RunNotAttributableError(RequestRejectedError)` with `__init__(self, request_id: str)` (403, the code, the message). Define `async def attributed_run_request(request: Request, data: Annotated[RunRequestModel, Depends(bound_run_request)], x_project_uuid: Annotated[str, Header()]) -> AttributedRun`. It calls `read_user_email(request.headers.get("Authorization"))`. On `None` it generates `request_id = str(uuid4())`, logs `logger.warning(format_log_event(<event>, {"header_project_uuid": x_project_uuid, "endpoint": request.url.path, "request_id": request_id}))` and raises `RunNotAttributableError(request_id)`. Otherwise it returns `AttributedRun(request=data, authorized_project_uuid=str(UUID(x_project_uuid)), user_email=email)`. `UUID(...)` can't fail here because the header already equals the raw body, which passed `UUID4` validation (R6). Because it depends on `bound_run_request`, a mismatch wins over a missing identity and an invalid body still returns 422 (R2). Satisfies FR-010, FR-011.
- [ ] T018 [US2] In `run_test` in `app/api/v1/routers/runs.py`, replace `data: Annotated[RunRequestModel, Depends(bound_run_request)]` with `run: Annotated[AttributedRun, Depends(attributed_run_request)]`. Every existing use of `data` in `run_test` now reads `run.request`, with no other behavior change. Remove the now-unused `bound_run_request` import. Satisfies FR-011 for tool and active runs, including all-`auth_token` and zero-test-case runs. Depends on T017.
- [ ] T019 [US2] Add a `TestRunAttribution` class with flow tests to `app/api/v1/routers/tests/test_runs.py`, with the same patches as T012. All tests use matching header and body unless stated. (1) A tool run whose `Authorization` is `"Bearer test-token"` (not a JWT), `make_cli_bearer_token(None)`, or `make_cli_bearer_token("")` (parametrized) returns 403 with a body exactly `{"message": RUN_NOT_ATTRIBUTABLE_MESSAGE, "data": None, "success": False, "code": "RUN_NOT_ATTRIBUTABLE", "request_id": <uuid>}`. `AWSLambdaClient` is never instantiated, `process_tool` is never called, nothing is minted, and there is exactly one `event=run_not_attributable` record whose `request_id` equals the response's (US2-3, FR-011). (2) An active run where every test case's `project` carries `auth_token` and the bearer has no `email` returns 403 `RUN_NOT_ATTRIBUTABLE`, with `ActiveAgentProcessor` never called and exactly one `event=run_not_attributable` record (FR-011). (3) A tool run with `test_definition` `{"tests": {}}` and no `email` returns 403 `RUN_NOT_ATTRIBUTABLE`, with exactly one `event=run_not_attributable` record (edge case "zero test cases"). (4) Header A, body B and a non-JWT bearer return 403 `PROJECT_MISMATCH`, with no `run_not_attributable` record and no `user_email` in the mismatch event (evaluation order in contracts/http-responses.md). (5) An invalid body `project_uuid` with a non-JWT bearer returns 422 with neither event (R2). (6) The bearer string doesn't appear in `caplog.text` in any of these tests (FR-014). Depends on T018.
- [ ] T020 [US2] Run `poetry run pytest -q`, `poetry run ruff check .` and `poetry run mypy .`, all green. Then commit T016–T019 with `feat: require user identity for runs`.

### Commit 4: `feat: mint run tokens from authorized project` (original plan #5)

- [ ] T021 [P] [US2] Add `generate_rsa_key_pair() -> tuple[str, str]` to `app/tests/utils.py`, returning `(private_pem, public_pem)`. Generate a 2048-bit RSA key with public exponent 65537, serialize the private key as PEM PKCS8 with no encryption and the public key as PEM SubjectPublicKeyInfo, and decode both to `str`, exactly as the `rsa_key_pair` fixture in `app/services/tests/test_jwt_generator.py` does. Leave that fixture untouched (Constitution XVI). `cryptography` is already a dependency.
- [ ] T022 [US2] Write unit tests for `RunTokenIssuer` in `app/services/runs/tests/test_token_issuer.py`. Patch `settings.JWT_SECRET_KEY` with `mocker.patch.object(settings, "JWT_SECRET_KEY", private_pem)`, using `generate_rsa_key_pair()`. Cases: (a) a passive issuer's `issue()` returns a token that decodes with the public key (`algorithms=["RS256"]`) to a payload whose keys are exactly `{"project_uuid", "exp", "iat"}`, with `project_uuid == authorized_project_uuid` and `exp - iat == DEFAULT_EXPIRATION_MINUTES * 60` (FR-004, FR-006, contracts/run-token.md). (b) Each call writes exactly one INFO record starting with `event=run_token_minted`, with fields in contract order: `user_email`, `project_uuid`, `agent_key`, `tool_key`, `run_type="passive"`, `request_id` (FR-012), and `project_uuid` equal to the decoded token's `project_uuid` claim. (c) An active issuer (`tool_key=None`, `run_type="active"`) writes a record without a `tool_key` field. (d) Two calls produce two records. (e) Neither the returned token nor any part of it appears in `caplog.text` (FR-014, SC-005). Depends on T021.
- [ ] T023 [P] [US2] Create `app/services/runs/token_issuer.py` with the module constant for the event name `run_token_minted` and the class `RunTokenIssuer`, constructed with keyword fields `authorized_project_uuid: str`, `user_email: str`, `agent_key: str`, `tool_key: str | None`, `run_type: RunType` and `request_id: str` (data-model.md "Run token issuer"), where `RunType = Literal["passive", "active"]` is a module-level alias, so no `def` line contains `pass` (`.coveragerc` excludes such lines along with the whole statement). `issue() -> str` calls `generate_jwt_token(self.authorized_project_uuid, settings.JWT_SECRET_KEY)`, imported at module level from `app.services.jwt_generator` so tests patch `app.services.runs.token_issuer.generate_jwt_token`. It then logs `logger.info(format_log_event(<event>, {"user_email": ..., "project_uuid": ..., "agent_key": ..., "tool_key": ..., "run_type": ..., "request_id": ...}))`, which omits `tool_key` when it is `None`, and returns the token. It never logs the token. Satisfies FR-004, FR-006, FR-012, FR-014.
- [ ] T024 [P] [US2] In `run` in `app/services/runs/tool_strategy.py`, add a `token_issuer: RunTokenIssuer` parameter. Replace `token = generate_jwt_token(str(data.project_uuid), settings.JWT_SECRET_KEY)` with `token = token_issuer.issue()`, keeping the per-test-case call and the `project[JWT_PROJECT_KEY] = token` injection. Remove the now-unused `generate_jwt_token` and `settings` imports; keep `JWT_PROJECT_KEY`. Nothing else changes; `process_tool` still receives `str(data.project_uuid)`. Satisfies FR-004, FR-012. Depends on T023.
- [ ] T025 [P] [US2] In `app/services/runs/active_strategy.py`, change `build_active_test_event(project_uuid, test_data, fallback_credentials=None)` to `build_active_test_event(test_data, token_issuer: RunTokenIssuer, fallback_credentials=None)`. Inside the unchanged `if JWT_PROJECT_KEY not in project:` guard, set `project[JWT_PROJECT_KEY] = token_issuer.issue()`. Add a `token_issuer: RunTokenIssuer` parameter to `run` and pass it as `build_active_test_event(test_data=test_data, token_issuer=token_issuer, fallback_credentials=fallback_credentials)`. Remove the now-unused `generate_jwt_token` and `settings` imports; keep `JWT_PROJECT_KEY`. Satisfies FR-004, FR-007, FR-012. Depends on T023.
- [ ] T026 [US2] In `run_test` in `app/api/v1/routers/runs.py`, after `request_id` is set and before branching on type, build `token_issuer = RunTokenIssuer(authorized_project_uuid=run.authorized_project_uuid, user_email=run.user_email, agent_key=run.request.agent_key, tool_key=None if run.request.type == "active" else run.request.tool_key, run_type=run.request.type, request_id=request_id)`. `tool_key` is set for passive runs only (data-model.md); the condition is written without the `passive` literal, because `.coveragerc` excludes every statement containing `pass`. Pass `token_issuer=token_issuer` to both `active_strategy.run(...)` and `tool_strategy.run(...)`. Nothing else changes. Satisfies FR-004, FR-012. Depends on T023, T024, T025.
- [ ] T027 [P] [US2] Update `TestBuildActiveTestEvent` in `app/services/runs/tests/test_active_strategy.py`. Replace the `project_uuid=` argument with `token_issuer=` set to a real `RunTokenIssuer(run_type="active", tool_key=None, ...)`, and replace the `mocker.patch.object(active_strategy, "generate_jwt_token", ...)` patches with `mocker.patch("app.services.runs.token_issuer.generate_jwt_token", return_value=...)`. `test_injects_jwt_when_missing` also asserts exactly one `event=run_token_minted` record. `test_preserves_existing_auth_token` asserts the token is kept, the mint mock isn't called, and there is no `run_token_minted` record (FR-007, US2-2). `test_falls_back_to_credentials_argument` and `test_parses_string_project_field` only swap the argument. Leave `TestBuildAgentResource` unchanged. Depends on T025.
- [ ] T028 [US2] In `app/api/v1/routers/tests/test_runs.py`, move every patch of `app.services.runs.tool_strategy.generate_jwt_token` and `app.services.runs.active_strategy.generate_jwt_token` to `app.services.runs.token_issuer.generate_jwt_token`. That covers `test_jwt_always_injected_in_credentials`, `mock_active_success_dependencies`, and the no-mint assertions in `TestRunProjectBinding` (T012) and `TestRunAttribution` (T019). Add a "no `event=run_token_minted` record" assertion to each of those rejection tests. Depends on T026.
- [ ] T029 [US2] Add a `TestRunTokenMinting` class with flow tests to `app/api/v1/routers/tests/test_runs.py`. Patch `settings.JWT_SECRET_KEY` with `generate_rsa_key_pair()`'s private key, and don't patch `generate_jwt_token`. (1) A tool run with two test cases streams 200 with two `TEST_CASE_COMPLETED` messages. Each `invoke_function` event's `sessionAttributes.project` `auth_token` decodes with the public key to exactly `project_uuid`/`exp`/`iat`, with `project_uuid == TEST_PROJECT_UUID`. There are exactly two INFO `run_token_minted` records, each with `user_email == TEST_USER_EMAIL`, `project_uuid` equal to the decoded token's claim, `agent_key`, `tool_key == TEST_TOOL_KEY`, `run_type == "passive"`, and `request_id` equal to the streamed messages' `request_id` (US2-1, US2-4, FR-006, FR-012, SC-002, SC-003). (2) With header and body both `TEST_PROJECT_UUID.upper()`, the minted token's `project_uuid` is the lower-case canonical value, and the `run_token_minted` record's `project_uuid` is that same lower-case value (R6, FR-004). (3) An active run with two test cases: one carries `project.auth_token = "user-supplied-token"`, the other has none. The first is passed through unchanged. The second gets a token whose `project_uuid == TEST_PROJECT_UUID`, not the test definition's `project.uuid` (FR-004). There is exactly one `run_token_minted` record, with `run_type == "active"` and no `tool_key` field (US2-2, US2-4, FR-007, edge case "mixed test cases"). (4) A tool run with `{"tests": {}}` and a valid identity streams 200 with zero `run_token_minted` records (edge case "zero test cases"). (5) With `invoke_function` raising on its first call, exactly one `run_token_minted` record exists (the token minted before the failure). With `process_tool` raising, there are none (edge case "identity readable but the run fails later"). (6) In every test, neither any minted token nor the bearer string appears in `caplog.text` (FR-014, SC-005). Depends on T021, T026.
- [ ] T030 [US2] Run `poetry run pytest -q`, `poetry run ruff check .` and `poetry run mypy .`, all green. Then commit T021–T029 with `feat: mint run tokens from authorized project`.

**Checkpoint**: Runs are bound, attributed and audited. The token contract is unchanged.

---

## Phase 5: User Story 3 – Other project-scoped CLI operations reject mismatched projects (Priority: P2)

**Goal**: Agents push, channel creation and ticketer creation reject a header/body project mismatch with the same 403 `PROJECT_MISMATCH` and event, before any configurator or `FlowsClient` call. `/permissions/verify` and the nested ticketer project keep today's behavior.

**Independent Test**: For each of the three endpoints, a matching request behaves as today and a mismatched one gets 403 plus one event, with no downstream call.

**Depends on**: Foundational, and `ensure_body_project_is_authorized` in `app/api/v1/project_binding.py` (T013, US1).

**Commit**: commit 5, `feat: bind project on push, channel, ticketer`.

### Tests for User Story 3

- [ ] T031 [P] [US3] Add flow tests to `app/api/v1/routers/tests/test_agents.py`, using `mock_auth_middleware`, `mock_helper_functions` and `mock_passive_configurator`. (1) With header `TEST_PROJECT_UUID` and a different body UUID, the response is 403 with the exact `PROJECT_MISMATCH` body, neither UUID in `response.text`, `configure_agents` never called, and exactly one `event=project_mismatch_rejected` record with `endpoint == api_path`. That record has no `user_email`, because `TEST_TOKEN` isn't a JWT and agents push doesn't require identity (US3-1, FR-001, FR-013). (2) A body of `TEST_PROJECT_UUID.upper()` returns 403 `PROJECT_MISMATCH`. (3) A body `project_uuid` of `not-a-uuid`, and a body without `project_uuid`, each with header `TEST_PROJECT_UUID`, return 422 with no mismatch record (edge case, 422 wins). (4) The bearer string isn't in `caplog.text` (FR-014). Existing tests stay unchanged.
- [ ] T032 [P] [US3] Add flow tests to `app/api/v1/routers/tests/test_channels.py`. Patch `app.api.v1.routers.channels.FlowsClient` and keep the class mock so constructor calls can be asserted. (1) With header = `project_uuid` fixture and a different body UUID, the response is 403 with the exact `PROJECT_MISMATCH` body, the `FlowsClient` class is never called, `create_channel` is never called, and there is exactly one `event=project_mismatch_rejected` record with `endpoint == api_path` (US3-2). (2) A case-only difference returns 403. (3) An invalid body UUID and a missing body `project_uuid`, each with a different header, return 422 with no mismatch record (edge case). (4) The bearer string isn't in `caplog.text`. Existing tests stay unchanged.
- [ ] T033 [P] [US3] Add flow tests to `app/api/v1/routers/tests/test_ticketers.py`, with the class-mock patch of `app.api.v1.routers.ticketers.FlowsClient`. (1) Header A and body B return 403 with the exact `PROJECT_MISMATCH` body, no `FlowsClient` construction, no `create_ticketer` call, and one event with `endpoint == api_path` (US3-2). (2) A case-only difference returns 403. (3) A top-level `project_uuid` matching the header with a different `ticketer_definition.config.project_uuid` returns 201 from the mocked Flows response. `create_ticketer` is called once with the definition, nested value unchanged, and no mismatch record exists (US3-3, FR-005). (4) An invalid body UUID and a missing body `project_uuid`, each with a different header, return 422 with no mismatch record (edge case). (5) The bearer string isn't in `caplog.text`.
- [ ] T034 [P] [US3] Add a flow test to `app/api/v1/routers/tests/test_permissions.py`. POST `/permissions/verify` with a random body `project_uuid`, an `Authorization` header, `X-CLI-Version: settings.CLI_MINIMUM_VERSION` and no `X-Project-Uuid`, using the existing `mock_connect_client` (HTTP 200). It returns 200 `{"status": "ok"}`, `ConnectClient` is constructed with the body project, and there is no `project_mismatch_rejected` record (US3-4, FR-005). This proves today's behavior; `permissions.py` isn't modified.

### Implementation for User Story 3

- [ ] T035 [US3] Add three dependencies to `app/api/v1/project_binding.py`, each delegating to `ensure_body_project_is_authorized` and returning the validated model. `async def bound_agents_request(request: Request, data: Annotated[ConfigureAgentsRequestModel, Form()], x_project_uuid: Annotated[str, Header()]) -> ConfigureAgentsRequestModel` reads the raw value from `(await request.form())["project_uuid"]`. `async def bound_channel_request(request: Request, data: CreateChannelRequestModel, x_project_uuid: Annotated[str, Header()]) -> CreateChannelRequestModel` and `async def bound_ticketer_request(request: Request, data: CreateTicketerRequestModel, x_project_uuid: Annotated[str, Header()]) -> CreateTicketerRequestModel` read it from `(await request.json())["project_uuid"]`; Starlette caches the body bytes. Only the top-level `project_uuid` is read; `ticketer_definition` is never inspected (FR-005). The validated model keeps today's 422 priority (R2). Satisfies FR-001, FR-002 for agents push, channels and ticketers.
- [ ] T036 [P] [US3] In `configure_agents` in `app/api/v1/routers/agents.py`, replace `data: Annotated[ConfigureAgentsRequestModel, Form()]` with `data: Annotated[ConfigureAgentsRequestModel, Depends(bound_agents_request)]`. Import `Depends` and `bound_agents_request`, and drop `Form` if unused. Nothing else changes. Depends on T035.
- [ ] T037 [P] [US3] In `create_channel` in `app/api/v1/routers/channels.py`, replace `data: CreateChannelRequestModel` with `data: Annotated[CreateChannelRequestModel, Depends(bound_channel_request)]`, importing `Depends` and `bound_channel_request`. The existing `x_project_uuid` parameter and the body stay unchanged. Depends on T035.
- [ ] T038 [P] [US3] In `create_ticketer` in `app/api/v1/routers/ticketers.py`, replace `data: CreateTicketerRequestModel` with `data: Annotated[CreateTicketerRequestModel, Depends(bound_ticketer_request)]`, importing `Depends` and `bound_ticketer_request`. The existing `x_project_uuid` parameter and the body stay unchanged. Depends on T035.
- [ ] T039 [US3] Run `poetry run pytest -q`, `poetry run ruff check .` and `poetry run mypy .`, all green, including every pre-existing agents, channels, ticketers and permissions test. Then commit T031–T038 with `feat: bind project on push, channel, ticketer`.

**Checkpoint**: All four covered endpoints share one binding rule.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Commits**: commit 6, `docs: fix run token lifetime in docstring` (T040), then commit 7, `docs: add 1.15.1 security changelog entry` (T041).

- [ ] T040 [P] In the `generate_jwt_token` docstring in `app/services/jwt_generator.py`, change "If not provided, uses default (60 minutes)." to "If not provided, uses default (2 minutes).". Change nothing else in the file (FR-008). Before committing, run `poetry run pytest -q`, `poetry run ruff check .` and `poetry run mypy .`. Commit with `docs: fix run token lifetime in docstring`.
- [ ] T041 [P] In `CHANGELOG.md`, insert a new entry directly above `## [1.15.0] - 2026-07-09`. The heading is `## [1.15.1] - 2026-10-05`, followed by a `### Security` subsection with three bullets. (1) `POST /api/v1/runs`, `/api/v1/agents`, `/api/v1/channels` and `/api/v1/ticketers` reject with HTTP 403 `PROJECT_MISMATCH` any request whose body `project_uuid` differs from the authorized `X-Project-Uuid` header, and run tokens are minted only from the authorized project. (2) Runs require a user identity (the bearer token's `email` claim) and otherwise fail with HTTP 403 `RUN_NOT_ATTRIBUTABLE`. (3) Each minted run token, rejected mismatch and unattributable run is recorded as a structured log line (`run_token_minted`, `project_mismatch_rejected`, `run_not_attributable`) that never contains token values. Don't reformat earlier entries (Constitution VIII, XVI). Commit with `docs: add 1.15.1 security changelog entry`.
- [ ] T042 Run the quality gate from the repository root: `poetry run pytest --cov-branch`, `poetry run ruff check .`, `poetry run mypy .`. All must be green. In the `--cov-report=term-missing` output, `app/core/log_events.py`, `app/api/v1/rejections.py`, `app/api/v1/user_identity.py`, `app/api/v1/project_binding.py`, `app/api/v1/run_attribution.py` and `app/services/runs/token_issuer.py` must show `Miss` 0 and `BrPart` 0; add tests for any gap (Constitution XIV, quickstart). Because `.coveragerc` excludes every line matching `pass`, confirm that `rg -n pass` on those six modules matches only the `RunType` alias, and that the `RunTokenIssuer(...)` statement in `app/api/v1/routers/runs.py` contains no `pass`. Confirm with `git diff --stat main...HEAD` that only the files listed in plan.md (plus this spec folder) changed, and that `app/api/v1/middlewares.py`, `app/api/v1/routers/permissions.py`, `app/api/v1/models/requests.py`, `app/clients/flows_client.py`, the agent configurators, `app/api/v1/middlewares_test.py` and `app/services/tests/test_jwt_generator.py` are untouched (Constitution XVI). Confirm that `git diff main...HEAD -- app/services/jwt_generator.py` changes only the docstring line (FR-008).

---

## Quickstart scenario coverage

| Quickstart row | Test tasks |
|---|---|
| US1-1 tool run, header A / body B | T012 (1), T028 |
| US1-2 active run, header A / body B | T012 (2), T028 |
| US1-3 same UUID, different case | T012 (3), T011 (b) |
| US1-4 mismatch event content | T012 (4), T011 (d)(e) |
| US2-1 legitimate tool run, N test cases | T029 (1), T022 (a)(b) |
| US2-2 active test case with its own `auth_token` | T029 (3), T027 |
| US2-3 identity unreadable (non-JWT, no `email`, empty `email`; all-`auth_token`; zero test cases) | T019 (1)(2)(3), T016 (c)(d) |
| US2-4 audit content | T029 (1)(3), T022 (b)(c) |
| US3-1/2 agents push, channel, ticketer mismatch | T031 (1), T032 (1), T033 (1) |
| US3-3 ticketer with a different nested project | T033 (3) |
| US3-4 `/permissions/verify` | T034 |
| Edge: invalid or missing body UUID with a different header → 422, no event | T012 (5), T019 (5), T031 (3), T032 (3), T033 (4) |
| FR-014 / SC-005: no minted token or bearer string in logs | T011 (g), T012 (6), T016 (e), T019 (6), T022 (e), T029 (6), T031 (4), T032 (4), T033 (5) |

Additional spec items: FR-004 (token from the header, never the body or test definition) is covered by T029 (2)(3). FR-006 is covered by T022 (a) and T029 (1). The mismatch-over-identity order is covered by T019 (4). The edge case "identity readable but the run fails later" is covered by T029 (5). The edge case "mismatch combined with other invalid input" is covered by T012 (1)(2)(5)(7).

Items with no task: FR-009, because the roles live in `AuthorizationMiddleware` (`ACCEPTABLE_ROLES`), which T042 confirms is untouched and whose role check `app/api/v1/middlewares_test.py` already tests. FR-015, because the token contract is unchanged (T022 (a), T029 (1)), conforming CLIs send the same project in header and body, and the CLI renders both new 403s from `message` (contracts/http-responses.md). The edge cases "header missing" and "header project not authorized in Connect", because `AuthorizationMiddleware` is unchanged and the binding runs only after it. Header missing is already tested (`test_runs.py` `missing_authorization`, `test_agents.py` `missing_project_uuid`); Connect's non-200 → 401 path has no test and is out of this feature's scope. The edge case "older CLI versions", because it is an assumption verified in the `weni-cli` source (spec.md, Assumptions) and can't be tested here.

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (Phase 1)**: none.
- **Foundational (Phase 2)**: after Setup. Blocks all stories.
- **US1 (Phase 3)**: after Foundational.
- **US2 (Phase 4)**: after US1. `attributed_run_request` depends on `bound_run_request`, and the T010 fixtures are required.
- **US3 (Phase 5)**: after US1 (reuses `ensure_body_project_is_authorized`). It doesn't depend on US2's code, but the commit order puts it after commit 4, so implement it after US2.
- **Polish (Phase 6)**: T040 and T041 are independent of code and can be prepared anytime, but they are committed last, in the order shown. T042 runs after everything else.

### Within phases

- T006 after T005. T008 after T007.
- T012 after T010 (same file). T013 after T003, T005, T009. T014 after T013.
- T018 after T017. T019 after T018.
- T022 after T021. T024 and T025 after T023. T026 after T023–T025. T027 after T025. T028 and T029 after T026 (same file, run sequentially). T029 also after T021.
- T036–T038 after T035.

---

## Parallel Examples

```text
# Phase 2, after T001:
T002 test_log_events.py | T003 log_events.py | T004 test_rejections.py | T005 rejections.py | T007 utils.py (make_cli_bearer_token) | T009 user_identity.py
# then T006 (main.py) and T008 (test_user_identity.py)

# Phase 3:
T010 test_runs.py fixtures | T011 test_project_binding.py | T013 project_binding.py
# then T012 (test_runs.py) and T014 (runs.py)

# Phase 4, commit 3:
T016 test_run_attribution.py | T017 run_attribution.py

# Phase 4, commit 4:
T021 utils.py (generate_rsa_key_pair) | T023 token_issuer.py
# then T024 tool_strategy.py | T025 active_strategy.py | T022 test_token_issuer.py
# then T026 runs.py | T027 test_active_strategy.py

# Phase 5:
T031 test_agents.py | T032 test_channels.py | T033 test_ticketers.py | T034 test_permissions.py
# after T035: T036 agents.py | T037 channels.py | T038 ticketers.py
```

---

## Implementation Strategy

### MVP (User Story 1)

1. Phase 1 baseline (T001).
2. Phase 2 (T002–T009). Commit 1 lands here.
3. Phase 3 (T010–T015). Commit 2 closes C1: a cross-project run is refused before anything is built or minted.
4. Stop and validate with T012 and the existing run suite.

### Incremental delivery

1. MVP (commits 1–2).
2. US2: attribution (commit 3), then issuer and audit (commit 4). Legitimate runs are unchanged and every token is traceable.
3. US3: defense in depth on push, channels and ticketers (commit 5).
4. Polish: docstring (commit 6), changelog (commit 7), quality gate (T042).

---

## Notes

- Write each phase's tests before its implementation and confirm they fail for the expected reason.
- Comments in new code explain only constraints the code can't show: why there is no `status_code` attribute (T005), why raw strings are compared (T013), and why the JWT signature isn't verified (T009).
- Meaningful literals (messages, codes, event names, the `Bearer ` prefix) are named constants (Constitution XV).
- The optional staging check in quickstart.md section 2 isn't a task. It can run after deployment.
