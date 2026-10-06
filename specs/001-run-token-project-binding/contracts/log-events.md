# Contract: structured log events

**Feature**: `001-run-token-project-binding` | **Research**: [R3](../research.md#r3-structured-log-format-for-the-mint-audit-record-and-the-mismatch-event)

## Line format

Records go through the existing root formatter, so a line looks like:

```text
2026-10-05 19:10:36 WARNING  event=project_mismatch_rejected header_project_uuid="6f1c2c1e-..." body_project_uuid="0b7d2d77-..." endpoint="/api/v1/runs" request_id="..." user_email="dev@example.com"
```

Rules:

- The message starts with `event=<name>`, followed by fields in the order listed below.
- Every field value is a JSON string literal (double-quoted, with `"`, `\` and control characters escaped). This prevents client-supplied values from injecting fields or lines.
- A field marked optional is omitted when its value is unknown. It is never emitted as an empty or placeholder value.
- Values never include a run token, a bearer token or the `Authorization` header (FR-014).
- Query: `{service_name="cli-backend"} |= "event=<name>" | logfmt`.

## `run_token_minted`

Level INFO. Exactly one per minted run token (FR-012).

| Field | Required | Value |
|---|---|---|
| `user_email` | yes | `email` claim of the bearer token |
| `project_uuid` | yes | Canonical authorized project, which equals the token's `project_uuid` claim |
| `agent_key` | yes | `agent_key` of the run request |
| `tool_key` | tool runs only | `tool_key` of the run request; absent for active runs |
| `run_type` | yes | `passive` (tool run) or `active` (active-agent run), the request's `type` |
| `request_id` | yes | The run's `request_id`, the same value the CLI receives in streamed responses |

Not emitted for active test cases that carry their own `auth_token`, or for
runs with zero test cases.

## `project_mismatch_rejected`

Level WARNING. Exactly one per rejected mismatch (FR-013, SC-006).

| Field | Required | Value |
|---|---|---|
| `header_project_uuid` | yes | Raw `X-Project-Uuid` header (the authorized project) |
| `body_project_uuid` | yes | Raw body `project_uuid`, as sent |
| `endpoint` | yes | Request path, e.g. `/api/v1/runs` |
| `request_id` | yes | The rejection's `request_id`, the same value as in the 403 body |
| `user_email` | optional | Present when the identity can be read; the 403 is returned either way |

## `run_not_attributable`

Level WARNING. Exactly one per run rejected because the identity cannot be
resolved.

| Field | Required | Value |
|---|---|---|
| `header_project_uuid` | yes | Raw `X-Project-Uuid` header (the authorized project) |
| `endpoint` | yes | Request path |
| `request_id` | yes | The rejection's `request_id`, the same value as in the 403 body |
