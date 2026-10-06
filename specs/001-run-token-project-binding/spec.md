# Feature Specification: Bind CLI Run Tokens to the Authorized Project

**Feature Branch**: `001-run-token-project-binding`

**Created**: 2026-10-05

**Status**: Draft

**Input**: User description: "Fix finding C1 from an internal security review: cross-project token minting in tool/agent runs (weni-cli-backend → retail-setup). The other findings of that review are not addressed in this spec."

## Background

The run endpoint authorizes the caller against Connect using the project in the request header, but mints the mesh token injected into the user's tool or agent using the project in the request body. Nothing compares the two. A user with a contributor (or higher) role in any project can therefore obtain a valid token for another customer's project. Their own code receives that token during the run and can use it against retail-setup, which accepts it and acts on the victim project's VTEX store: orders, customer personal data, and the card-related payment fields VTEX exposes. The token's 2-minute lifetime does not mitigate this, because the attacker's code uses it immediately and runs can be repeated at will.

## Clarifications

### Session 2026-10-05

- Q: Where does this spec live, and what happens to the other findings of the review? → A: In `weni-cli-backend`, because every change is in this service. Only C1 is in scope; the other findings are handled separately.
- Q: How is the project inside a minted run token tied to the project the user was authorized for? → A: The request is rejected when the project in the request body differs from the authorized project (the project identified in the request header and checked against Connect). When they match, the token is minted only from the authorized header value.
- Q: Which CLI backend endpoints get the header/body project check? → A: Every authenticated endpoint that takes a project identifier at the top level of its body: runs (tool and active-agent), agents push, channel creation and ticketer creation.
- Q: Which response does a mismatch produce? → A: HTTP 403 Forbidden.
- Q: Does the run token gain new claims (audience, issuer, subject, unique id)? → A: No. The token payload stays exactly as it is today (project identifier, expiry, issued-at).
- Q: Does retail-setup change in this spec? → A: No. C1 is fixed entirely in the CLI backend.
- Q: Should each token mint be audited? → A: Yes, as a structured application log record that never contains the token value.
- Q: Should rejected mismatches be recorded? → A: Yes, as a warning-level structured security event with the header project, body project, endpoint and request id.
- Q: Is investigating past exploitation part of this spec? → A: No, it is handled separately.
- Q: Active-agent test cases may carry their own `auth_token`, and minting is skipped for them. Keep that? → A: Keep it. The user already holds that token, so no privilege is gained.
- Q: Do token lifetime or the roles allowed to run change? → A: No. Lifetime stays at 2 minutes and roles stay contributor, moderator and support. Only the documentation that wrongly states a 60-minute lifetime is corrected.
- Q: Where does the user identity in the mint audit record come from? → A: From the user's bearer credential that the CLI backend already receives, with no extra call.
- Q: Is the permissions verification endpoint (used by `weni project use`) included? → A: No. It does not require the project header, and its purpose is to verify access to the project in the body.
- Q: Is the nested project identifier inside a ticketer definition's configuration checked? → A: No. Only the top-level project identifier of each request body is in scope.
- Q: What if the user identity can't be read from the bearer credential? → A: The run fails. Identity is required for every run, including active-agent runs where every test case supplies its own token.
- Q: How are header and body project identifiers compared? → A: As exact strings. Any textual difference, including letter case, is a mismatch.
- Q: Does the mismatch security event include the user identity? → A: Yes, when it can be read. The 403 is returned whether or not the identity is available.
- Q: What does the 403 response body say? → A: That the project in the request does not match the authorized project, without echoing either project identifier.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A user cannot obtain a token for a project they are not authorized in (Priority: P1)

A Weni user with a contributor (or higher) role in project A uses `weni run`, or calls the CLI backend's run endpoint directly. They send project A in the header, which is what gets authorized, and project B, which belongs to another customer, in the body. Today the backend mints a mesh token for project B and hands it to code the user wrote, which can then read and write project B's VTEX orders and payment data through retail-setup. After this change the request is refused before any tool or agent is built or executed, no token is minted, and the attempt is recorded as a security event.

**Why this priority**: This is the critical finding. It is the entry point of the attack chain that exposes every connected store's order, card-related and personal data.

**Independent Test**: With a user who is authorized in project A only, submit a tool run and an active-agent run where the header carries A and the body carries B. Verify that each request gets HTTP 403, that no execution environment is created, that no token is minted, and that one mismatch security event exists for each attempt.

**Acceptance Scenarios**:

1. **Given** a user authorized in project A, **When** they submit a tool run with header project A and body project B, **Then** the response is HTTP 403 with a message saying the request's project does not match the authorized project, the message contains neither project identifier, no tool is packaged or deployed, and no token is minted.
2. **Given** the same user, **When** they submit an active-agent run with header project A and body project B, **Then** the outcome matches scenario 1.
3. **Given** the same user, **When** the header and body carry the same UUID but in different letter case, **Then** the request is rejected as a mismatch (exact comparison).
4. **Given** any rejected mismatch, **When** operators look at the CLI backend logs, **Then** they find one warning-level security event with the header project, body project, endpoint, request id and, when it can be read, the user identity.

---

### User Story 2 - Legitimate runs keep working unchanged (Priority: P1)

A developer runs `weni run` for a tool or an active agent in the project selected with `weni project use`. The CLI sends that project in both the header and the body, as it does today. The run behaves exactly as before. The token injected into the tool or agent has the same format, lifetime and claims as today, and carries the authorized project. Each mint leaves an audit record that ties the token to the user who started the run.

**Why this priority**: The fix must not break the development workflow of every agent builder, and the token contract that retail-setup and the tools rely on must stay the same.

**Independent Test**: With the current CLI, run an existing tool test definition and an existing active-agent test definition against a project the user is authorized in. Verify that the results match a run before the change, that the injected token decodes to the same three claims with the authorized project, and that one audit record exists per minted token.

**Acceptance Scenarios**:

1. **Given** a user authorized in project A, **When** they run a tool test definition with N test cases and header and body both set to A, **Then** every test case executes, each receives a token for project A with the same claims and lifetime as today, and N mint audit records exist.
2. **Given** an active-agent test case whose project block already contains an `auth_token`, **When** the run executes, **Then** that token is passed through unchanged, no token is minted for that test case, and no mint audit record is written for it.
3. **Given** a valid run, **When** the bearer credential does not expose a readable user identity, **Then** the run fails before any tool or agent is built or executed, no token is minted, and the user gets an error saying the run could not be attributed to a user. This applies even when every test case supplies its own token.
4. **Given** any minted token, **When** its mint audit record is inspected, **Then** the record contains the user identity, authorized project, agent key, tool key (tool runs only), run type and request id, and does not contain the token value.

---

### User Story 3 - Other project-scoped CLI operations reject mismatched projects (Priority: P2)

The same header/body confusion exists on the other CLI backend operations that take a project in the body: agents push, channel creation and ticketer creation. These operations forward the user's own credential to downstream services that re-authorize, so they do not mint mesh tokens. Applying the same check everywhere gives one consistent rule: an operation always acts on the project the user was authorized for.

**Why this priority**: This is defense in depth. These paths are not exploitable today in the way C1 is, but the inconsistency is cheap to remove and avoids the same flaw reappearing when one of them starts minting tokens.

**Independent Test**: For each of agents push, channel creation and ticketer creation, send one request with matching projects and one with mismatched projects. Verify that matching requests behave as today and mismatched ones get HTTP 403 plus a security event, with no downstream call made.

**Acceptance Scenarios**:

1. **Given** a user authorized in project A, **When** they push agents with header A and body B, **Then** the response is HTTP 403, nothing is pushed, and a mismatch security event is recorded.
2. **Given** the same user, **When** they create a channel or a ticketer with header A and body B, **Then** the response is HTTP 403, no downstream call is made, and a mismatch security event is recorded.
3. **Given** a ticketer creation request whose top-level project is A (matching the header) but whose ticketer configuration carries a different nested project identifier, **When** the request is processed, **Then** it behaves as today. The nested value is not checked.
4. **Given** a request to the permissions verification endpoint, **When** it carries any project in the body and no project header, **Then** it behaves as today.

---

### Edge Cases

- **Header missing**: Today's behavior stays the same (the request is rejected before authorization). The mismatch rule never runs without an authorized header.
- **Body project missing or not a valid UUID**: Today's request validation error stays the same. It is not reported as a mismatch.
- **Header project not authorized in Connect**: Today's 401/403 from authorization stays the same. The mismatch check applies only to requests that passed authorization.
- **Mismatch combined with other invalid input**: The mismatch is detected before any tool, agent, channel or ticketer processing starts, so nothing is created, deployed or forwarded.
- **Run with zero test cases**: No token is minted and no mint audit record is written, but the identity requirement still applies to the run.
- **Active-agent run with mixed test cases** (some carry their own token, some don't): Tokens are minted and audited only for test cases without one.
- **Identity readable but the run fails later** (packaging, deployment, execution): Audit records exist only for tokens actually minted before the failure.
- **Older CLI versions**: Every CLI version that passes the minimum-version check sends the same stored project in both header and body, so legitimate users of those versions see no mismatch.

## Requirements *(mandatory)*

### Functional Requirements

**Project binding**

- **FR-001**: The CLI backend MUST reject with HTTP 403 any request to the run endpoint (tool and active-agent), agents push, channel creation and ticketer creation whose valid top-level body project identifier is not exactly equal, as a string, to the project identifier in the request header that authorization checked. A missing or invalid body project identifier keeps today's validation error (see Edge Cases).
- **FR-002**: The mismatch check MUST happen after the existing authorization succeeds and before any processing of the request (packaging, deploying or invoking tools and agents, or calling downstream services).
- **FR-003**: The 403 response body MUST say that the project in the request does not match the authorized project and MUST NOT include either project identifier.
- **FR-004**: When a run mints a token, the project in the token MUST come from the authorized header value, never from the request body or the test definition.
- **FR-005**: The permissions verification endpoint and the nested project identifier inside ticketer configurations MUST keep today's behavior.

**Token contract (unchanged)**

- **FR-006**: The minted run token MUST keep exactly today's claims (project identifier, expiry, issued-at), signing method and 2-minute lifetime. No claims are added or removed.
- **FR-007**: Active-agent test cases that already carry an `auth_token` in their project block MUST keep it unchanged, and no token is minted for them.
- **FR-008**: The documentation of the token generator MUST state the real default lifetime (2 minutes) instead of 60 minutes.
- **FR-009**: The roles allowed to start runs (contributor, moderator, support) MUST stay the same.

**Attribution and audit**

- **FR-010**: Every run MUST resolve the identity of the requesting user from the bearer credential the CLI backend already receives, without calling any additional service.
- **FR-011**: If the user identity cannot be resolved, the run MUST fail before any tool or agent is built or executed, MUST NOT mint any token, and MUST tell the user that the run could not be attributed to a user. This applies whether or not any test case would need a minted token.
- **FR-012**: Each minted token MUST produce exactly one structured application log record with: user identity, authorized project, agent key, tool key (tool runs only), run type and request id.
- **FR-013**: Each rejected mismatch MUST produce exactly one warning-level structured security event with: header project, body project, endpoint, request id and user identity when it can be resolved.
- **FR-014**: Token values MUST NOT appear in mint audit records, mismatch events or any other log line added by this feature.

**Boundaries**

- **FR-015**: retail-setup, agentic-cx, weni-cli and Nexus MUST NOT require changes for this feature to work.

### Key Entities

- **Authorized project**: The project identified in the request header and confirmed by Connect for the calling user and role. It is the only source of truth for which project a CLI operation acts on and which project a minted token carries.
- **Requested project**: The project identifier at the top level of the request body. It is accepted only when it exactly equals the authorized project.
- **Run token**: A short-lived mesh token minted per test case and injected into the user's tool or agent. Its contract stays the same.
- **Mint audit record**: A log record linking one minted token to the user, project, agent/tool, run type and request.
- **Mismatch security event**: A warning-level log record of a rejected attempt to act on a project other than the authorized one.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of requests whose body project differs from the authorized project are refused on the four covered operations, with zero tokens minted and zero tools, agents, channels or ticketers created as a result.
- **SC-002**: Every run token minted after release carries the authorized project. In a review of mint audit records, the token's project never differs from the authorized project.
- **SC-003**: 100% of minted tokens can be traced to the user who requested them from the mint audit records alone.
- **SC-004**: Existing tool and active-agent test definitions run with the current CLI produce the same results as before the change. No agent builder needs to change their CLI, definitions or tools.
- **SC-005**: No token value appears in any log record produced by this feature.
- **SC-006**: Operators can list every rejected cross-project attempt (who, which projects, which operation, when) from the security events alone.

## Assumptions

- Connect's authorization check on the header project stays the trust anchor for "the user may act on this project". This feature does not change it.
- The bearer credential received by the CLI backend carries a readable user identity for interactive CLI logins. The exact identity attribute (for example, e-mail or subject) is chosen in the plan phase.
- Structured application logs from the CLI backend already reach the team's log platform, and their retention is enough for security review. No new log destination is introduced.
- The CLI sends the same stored project in header and body for every covered operation (verified in the current `weni-cli` source).

## Out of Scope

- All other findings of the security review, including this service's error-reporting and cross-origin settings.
- Adding audience, issuer, subject or unique-id claims to any token, and any token validation change in retail-setup or agentic-cx.
- Investigating whether C1 was exploited before this fix.
- Changes to Nexus, which mints production tool tokens, and to payment-ms.
- Changes to the token lifetime or to the roles allowed to run, push or create resources.
