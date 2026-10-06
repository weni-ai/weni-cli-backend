# Quickstart: validating run-token project binding

**Feature**: `001-run-token-project-binding` | **Plan**: [plan.md](./plan.md)

This guide proves the feature works. Response shapes, log fields and token
claims are defined in [contracts/](./contracts/), and are not repeated here.

## Prerequisites

- Python 3.12 and the project's Poetry environment (`poetry install`). The
  repository's `pyproject.toml` uses `package-mode`, which needs Poetry 1.8 or
  later. With an older Poetry, run the same commands through the existing
  virtualenv's `python -m`.
- Automated checks need no network, AWS, Connect or Flows access: tests use
  `mock_auth_middleware`, patched Lambda/Flows/Nexus/Gallery clients, and a
  generated RSA key for `JWT_SECRET_KEY`.

## 1. Automated validation (required)

Run these while iterating:

```bash
poetry run pytest -q app/api/v1/tests app/services/runs/tests app/services/tests
poetry run pytest -q app/api/v1/routers/tests/test_runs.py app/api/v1/routers/tests/test_agents.py \
  app/api/v1/routers/tests/test_channels.py app/api/v1/routers/tests/test_ticketers.py \
  app/api/v1/routers/tests/test_permissions.py
```

Then run the gate before opening the PR (Constitution quality gates):

```bash
poetry run pytest --cov-branch
poetry run ruff check .
poetry run mypy .
```

Expected: everything is green, and branch coverage of the new modules leaves
no uncovered branch.

| Scenario (spec) | Proven by a flow test that checks |
|---|---|
| US1-1 tool run, header A / body B | 403 `PROJECT_MISMATCH`; no Lambda `create_function`, no `process_tool`, no mint; one `project_mismatch_rejected` record; body contains neither UUID |
| US1-2 active run, header A / body B | Same as US1-1, with `ActiveAgentProcessor` never called |
| US1-3 same UUID, different case | 403 `PROJECT_MISMATCH` |
| US1-4 mismatch event content | The record has `header_project_uuid`, `body_project_uuid`, `endpoint`, `request_id` and, when the token has `email`, `user_email`. `request_id` equals the response's |
| US2-1 legitimate tool run, N test cases | Streams as today; each injected `auth_token` decodes (with the test public key) to exactly `project_uuid`/`exp`/`iat` with the authorized project; N `run_token_minted` records |
| US2-2 active test case with its own `auth_token` | The token is passed through unchanged; no mint and no record for that case; mixed definitions mint only for the others |
| US2-3 identity unreadable (non-JWT bearer, no `email`, empty `email`) | 403 `RUN_NOT_ATTRIBUTABLE` before streaming; no Lambda, no mint; one `run_not_attributable` record; this also holds for an all-`auth_token` active run and for zero test cases |
| US2-4 audit content | The record has `user_email`, `project_uuid`, `agent_key`, `tool_key` (tool only), `run_type`, `request_id`; no token |
| US3-1/2 agents push, channel, ticketer mismatch | 403 `PROJECT_MISMATCH`; the configurator or `FlowsClient` is never called; one event each |
| US3-3 ticketer with a different nested project | Behaves as today (201 from the mocked Flows) |
| US3-4 `/permissions/verify` | Unchanged; no header required |
| Edge: invalid or missing body UUID with a different header | 422, as today; no mismatch event |
| FR-014 / SC-005 | Captured logs from every scenario above contain no minted token and no bearer string |

## 2. Manual end-to-end check (staging, optional)

Prerequisites: a staging deployment of this branch, a user who is a
contributor in project A but not in project B, the current `weni-cli`, and a
CLI login (`weni login`, then `weni project use <A>`).

1. **Legitimate run**: run `weni run` with an existing tool test definition, then with an active-agent definition. Expected: results match a run on the current release.
2. **Cross-project attempt**: replay the run request with `curl -X POST "$CLI_BACKEND/api/v1/runs" -H "Authorization: Bearer $TOKEN" -H "X-Project-Uuid: <A>" -H "X-CLI-Version: <current>" -F project_uuid=<B> ...`, using the same form fields the CLI sends. Expected: the 403 from [http-responses.md](./contracts/http-responses.md#403-project_mismatch), and no Lambda named `cli-*` is created.
3. **Logs**: in Grafana Explore, on the Loki datasource:
   - `{service_name="cli-backend"} |= "event=project_mismatch_rejected" | logfmt` shows one line for step 2, whose `request_id` equals the 403's.
   - `{service_name="cli-backend"} |= "event=run_token_minted" | logfmt | request_id="<run id from step 1>"` shows one line per test case that needed a token.
   - `{service_name="cli-backend"} |~ "event=(run_token_minted|project_mismatch_rejected|run_not_attributable)" |= "eyJ"` returns nothing (every JWT starts with `eyJ`).
