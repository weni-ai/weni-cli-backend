# Contract: HTTP responses of the covered endpoints

**Feature**: `001-run-token-project-binding` | **API**: `/api/v1` (unchanged prefix, Constitution IV)

Conforming clients (every CLI version that passes the minimum-version check)
send the same project in the `X-Project-Uuid` header and in the body, and a
Keycloak access token with an `email` claim. For them every response below is
unchanged. The only additions are two 403 responses for requests that today
succeed unsafely.

## Covered endpoints

| Method and path | Body type | Body project field | Binding check | Identity required |
|---|---|---|---|---|
| `POST /api/v1/runs` (tool and active) | multipart form | `project_uuid` | yes | yes |
| `POST /api/v1/agents` | multipart form | `project_uuid` | yes | no |
| `POST /api/v1/channels` | JSON | `project_uuid` | yes | no |
| `POST /api/v1/ticketers` | JSON | `project_uuid` (top level only) | yes | no |
| `POST /api/v1/permissions/verify` | JSON | `project_uuid` | **no** (FR-005) | no |

## Evaluation order

1. `VersionCheckMiddleware` and `AuthorizationMiddleware`: unchanged (426/400/401/403/500).
2. Body validation: unchanged 422 `{"detail": [...]}` when `project_uuid` is missing or not a UUID4, or when any other field is invalid.
3. Binding check: 403 `PROJECT_MISMATCH` (below).
4. Runs only, identity check: 403 `RUN_NOT_ATTRIBUTABLE` (below).
5. Endpoint body: unchanged behavior, including today's 400s (for example, a missing tool zip) and streaming responses.

## 403 PROJECT_MISMATCH

Returned when the raw body `project_uuid` is not exactly equal, as a string, to
the `X-Project-Uuid` header. Letter case and formatting count.

```json
{
  "message": "The project in the request does not match the authorized project.",
  "data": null,
  "success": false,
  "code": "PROJECT_MISMATCH",
  "request_id": "<uuid4 generated for this rejection>"
}
```

- `Content-Type: application/json`, not streamed.
- Neither project identifier appears anywhere in the body (FR-003).
- Side effects: none. No form is processed, no Lambda is created, no token is minted, and Nexus, Gallery and Flows are not called. Exactly one `project_mismatch_rejected` log event is written ([log-events.md](./log-events.md)).
- Not reported to Sentry.

## 403 RUN_NOT_ATTRIBUTABLE (runs only)

Returned when the projects match but the user identity cannot be read from the
`Authorization` header. That happens when the header is not `Bearer <jwt>`,
the JWT payload cannot be decoded, or `email` is missing, empty or not a
string. This applies to tool and active runs, including active runs whose test
cases all carry an `auth_token` and runs with zero test cases (FR-011).

```json
{
  "message": "This run could not be attributed to a user.",
  "data": null,
  "success": false,
  "code": "RUN_NOT_ATTRIBUTABLE",
  "request_id": "<uuid4 generated for this rejection>"
}
```

- Side effects: none, as for the mismatch response. Exactly one `run_not_attributable` log event is written.
- Not reported to Sentry.
- When the request also has a mismatch, the mismatch response wins (step 3 runs before step 4).

## CLI rendering (no CLI change required, FR-015)

- `weni run` and `weni project push` read `message`, and fall back to `detail`.
- `weni channel create` and `weni ticketer create` read `message`.
- 401 is reserved for authentication failures. The CLI turns any 401 into "please login again", which is why both rejections use 403.
