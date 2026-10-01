<!--
Sync Impact Report:
- Version change: none → 1.0.0
- Modified principles: template placeholders replaced by the root engineering
  constitution and the backend domain constitution, instantiated for this
  FastAPI service
- Added sections: Technology and Layout, Quality Gates
- Removed sections: none
- Follow-up TODOs:
  - TODO(SENTRY_PII): app/main.py initializes Sentry with send_default_pii=True.
    Observability and Diagnosable Errors forbid secrets and sensitive personal
    data on logs and error reports. This constitution does not waive that rule;
    the initialization must be reconciled in a later change.

Provenance:
- Source: weni-ai/vtex-cx-engineering-constitutions (main)
- Domains: backend
- Generated: 2026-10-01
-->

# Weni CLI Backend Constitution

## Core Principles

### I. Version Control and Review

All code MUST enter the default branch through a pull request. A merge MUST
require at least one approved review and a green CI run of the Lint, Test and
Coverage workflow. Direct pushes to the default branch MUST be blocked by
platform branch protection.

**Rationale:** The policy is only real when the platform enforces it. Peer
review and a protected default branch keep history auditable and stop
unreviewed changes from reaching production.

### II. Security and Secrets

Secrets MUST never be committed. Runtime secrets — including `SENTRY_DSN`,
`JWT_SECRET_KEY`, `ELASTIC_APM_SECRET_TOKEN`, and AWS credentials — MUST be
injected from the environment through `app/core/config.py` (`pydantic-settings`).
`.env` MUST stay untracked; `.env.example` MUST document keys without real
values. Access to Nexus, Connect, Gallery, Flows, and AWS MUST follow least
privilege. Dependencies MUST be declared in `pyproject.toml`, locked by Poetry,
and checked for known vulnerabilities. They MUST come only from trusted sources.

**Rationale:** Leaked credentials and untrusted dependencies are among the most
common and most damaging breaches. Prevention is cheaper than remediation.

### III. Observability

Logs MUST be structured and MUST never contain secrets or sensitive personal
data. Errors MUST be traceable across this service, its HTTP clients
(`app/clients/`), and AWS calls through a correlation or trace identifier.
`request_id` on CLI responses is the in-process correlation handle and MUST be
propagated when a call leaves the process. Sentry and, when configured, Elastic
APM are the error and trace backends for this service.

**Rationale:** Structured, privacy-safe telemetry is what makes incidents
diagnosable without creating a new data-exposure risk.

### IV. Versioned Contracts

The public HTTP API lives under `/v1` in `app/api/v1`. Any change to that
contract, or to another public interface this repository exposes, MUST follow
SemVer. Changes MUST stay backward compatible within the same major version or
ship with an announced deprecation path. A silent breaking change MUST NOT be
introduced. A new incompatible HTTP contract MUST be published as a new major
prefix rather than by mutating `/v1` in place.

**Rationale:** The Weni CLI and other callers depend on a stable contract.
Explicit versioning and deprecation give them a predictable path to adapt
without outages.

### V. Specification Traceability

Every engineering spec MUST derive from exactly one approved product spec and
MUST reference it through an immutable, pinned version (commit or tag). A
mutable URL or ID alone MUST NOT be used. The product spec MUST exist and be
tagged before its engineering spec is created. An engineering spec MUST NOT
redefine the "what" it inherits: problem, scope, success criteria, and binding
decisions belong to the product spec. A technical architecture document SHOULD
be produced for non-trivial features; when it exists it MUST be linked from the
engineering spec, also pinned by commit or tag, but its absence MUST NOT block
the engineering spec.

Every engineering spec MUST open with an inheritance section in exactly this
format:

```
## Inheritance from Product Spec
- Product Spec: <title> — <URL>
- Pinned version: <commit/tag>
- Architecture doc: <none | URL + commit/tag>
- Inherited binding decisions: <short list>
- Scope of this spec: <slice implemented by this repo>
- Divergences: <none | link to amendment>
```

**Rationale:** Traceability from product intent to technical execution keeps
decisions auditable. Pinning the version guarantees that every team implements
the same version of the feature. A mandatory product spec prevents engineering
work without an agreed problem. An optional architecture document avoids
blocking delivery when the design is trivial. A single inheritance format keeps
the link machine-checkable.

### VI. No Silent Divergence

When a technical need contradicts something inherited from the product spec —
scope, success criteria, or a binding decision — the divergence MUST NOT be
implemented silently in code. It MUST be raised as an amendment in the product
repository and recorded in the `Divergences` field of the engineering spec's
inheritance section, linking to that amendment. Once the amendment is approved
and produces a new tag, the engineering spec's `Pinned version` MUST be updated
to it. A technical difference that contradicts nothing inherited is an
implementation decision and MUST live in the engineering spec.

**Rationale:** In a federated model the product spec is the source of truth. A
silent code deviation separates intent from implementation with no audit trail.
Amendments keep the spec authoritative.

### VII. Commit Messages

Commits MUST follow Conventional Commits: `<type>: <description>`. Allowed
types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`. The description MUST
be imperative, specific, and no longer than 50 characters. Commits MUST be
atomic: one logical change per commit.

**Rationale:** Conventional commits enable changelog generation and semantic
versioning. Atomic commits simplify bisecting, reverting, and reviewing.

### VIII. Changelog Maintenance

Public libraries MUST maintain a changelog in Keep a Changelog format. This
service is not a published library; it MUST still maintain `CHANGELOG.md` in
that format because the Weni CLI consumes this API as a versioned contract.
Every user-facing change MUST appear under Added, Changed, Deprecated, Removed,
Fixed, or Security. Version bumps MUST follow SemVer.

**Rationale:** A maintained changelog communicates impact to consumers and
serves as release documentation. SemVer alignment keeps upgrade expectations
predictable.

### IX. Never Trust the Client

Everything that reaches this service from outside — the Weni CLI, another Weni
API, a webhook, or any other caller — MUST be treated as potentially malicious,
incomplete, or incorrect until it is validated. Every external input MUST be
validated for type, format, range, and business rules at the boundary, using
Pydantic models in `app/api/v1/models` and the routers in `app/api/v1/routers`,
before use. Authorization MUST be enforced on the server for every request
through `AuthorizationMiddleware`, regardless of any check already performed by
the client.

**Rationale:** Clients run outside this service's control and can be inspected,
modified, or bypassed. Server-side validation is what prevents injection, data
corruption, and privilege escalation.

### X. Fail Gracefully and Predictably

Calls to external dependencies — Nexus, Connect, Gallery, Flows, AWS Lambda,
and CloudWatch Logs — MUST have an explicit timeout and MUST NOT block
indefinitely. Failures MUST be handled explicitly and surfaced as consistent
error responses. They MUST NOT become unhandled crashes or leaked internal
details.

**Rationale:** Failure is certain. Explicit handling keeps a partial outage
contained and observable instead of taking down the process or exposing
internals to the CLI.

### XI. Bounded Retry Over REST

When data is propagated to another service over REST, a failure in that call
MUST be retried rather than dropped. A retry MUST be attempted only when the
failure could plausibly succeed on another attempt — a connection error, a
request timeout, an HTTP 5xx, or an HTTP 429 — and MUST NOT be attempted on a
4xx that reflects a defect in the request. A retry MUST apply only to an
operation that is idempotent or protected by a deduplication key; when the
operation is neither, it MUST be made idempotent rather than left without
retry. Every retry policy MUST define a maximum number of attempts and a
backoff strategy. Unbounded retry MUST NOT be used. When attempts are
exhausted, the failure MUST be logged and MUST remain recoverable. It MUST NOT
be silently discarded.

**Rationale:** Propagation fails for transient reasons more often than for
permanent ones, so retry is what keeps services converging. Retrying a rejected
request or a non-idempotent operation multiplies load or duplicates the effect.
Bounds keep the mechanism from becoming the outage. An observable, recoverable
exhausted case is what prevents data from disappearing between two services
that each believe they succeeded.

### XII. Scalability and Peak Load

This service MUST stay stateless so it can scale horizontally. State that
outlives a single request MUST NOT be kept in process memory or on local disk.
It MUST live in an external store shared by all instances. The peak load the
service is expected to sustain MUST be declared in its engineering spec, stated
as peak and not as average.

**Rationale:** Capacity is a design input. Sizing for average traffic fails
exactly when demand matters most. Statelessness is what makes adding instances
a valid answer to load. A declared peak turns scalability into a number that
can be reviewed and tested.

### XIII. Diagnosable Errors

Every error reported to Sentry or Elastic APM MUST carry enough context to be
located and filtered without reproducing it: at minimum the project identifier,
the account identifier, the user identifier, and the correlation identifier of
the request. Those identifiers MUST be opaque. Sensitive personal data — names,
e-mail addresses, phone numbers, or government identifiers — MUST NOT be
attached to an error report.

**Rationale:** An error without identifying context can be counted but not
investigated. Opaque identifiers give the filtering an investigation needs
while keeping the report free of personal data.

### XIV. Tests Exercise Flows

Every flow MUST have at least one test covering the complete use case, from
input to resulting effect. Tests that assert a single method in isolation are
allowed and SHOULD be used for edge cases and input variations that are
expensive to reach through the whole flow, but they MUST NOT be the only
coverage a flow has. Every flow MUST cover its success path and its failure
paths. An error path that no test exercises MUST NOT be considered covered.
Flow tests for HTTP behavior live beside the routers under
`app/api/v1/routers/tests`. Client and service tests live beside the code they
cover.

**Rationale:** A suite made only of isolated method tests can be green while
the composition of those methods is broken. Method-level tests remain the
cheapest way to cover many inputs. The flow test proves the pieces work
together. Failure paths are the least exercised in development and the most
expensive in production.

### XV. Explicit Over Clever

What a piece of code does MUST be evident where it happens. Hidden side effects
and implicit control flow MUST NOT be introduced to save lines. Any literal
that carries meaning — a threshold, a limit, a timeout, a retry count — MUST
be a named constant rather than an inline value. A literal that carries no
meaning beyond its own value, such as an index of 0 or an increment of 1, is
exempt. Comments MUST explain why a decision was made: the constraint, the
trade-off, or the non-obvious reason. A comment that restates what the code
already says is a signal that the code SHOULD be rewritten to say it.

**Rationale:** Code is read far more often than it is written, usually by
someone without the context that made the clever version feel obvious. An
unexplained literal is a decision nobody can review. Comments on the why
preserve the information the code cannot carry.

### XVI. Contained Changes

A change MUST be limited to the context it was asked to address. Refactoring,
renaming, reformatting, or behavior adjustments outside that context MUST NOT
ride along. Each belongs to its own change. This principle governs the scope of
a change as a whole. The requirement that each commit be atomic governs how
that change is divided internally. A change that stays within scope MAY still
span several commits.

**Rationale:** A change that reaches beyond its stated scope is a change nobody
reviewed on purpose. It hides the intended fix inside unrelated edits, makes
the diff expensive to read, and turns a revert into a choice between losing the
fix and keeping an unrelated regression.

## Technology and Layout

This repository is the FastAPI backend for the Weni CLI.

- Language and runtime: Python 3.12 (CI also runs 3.13). Dependencies are
  managed with Poetry (`pyproject.toml`, `poetry.lock`).
- HTTP: FastAPI and Pydantic v2. The application factory is `app/main.py`.
  Routers are mounted at `/v1` from `app/api/v1/routes.py`.
- Layout: `app/api/v1` for routers, models, and middlewares; `app/services` for
  use cases; `app/clients` for outbound HTTP and AWS; `app/core` for settings
  and shared response types.
- Outbound dependencies: Nexus, Connect (`WENI_API_URL`), Gallery, Flows, AWS
  Lambda, and CloudWatch Logs.
- Local run: `poetry run uvicorn app.main:app` or Docker Compose (`api` /
  `api-dev`). Container definition lives in `docker/Dockerfile`.

## Quality Gates

A change MUST pass, locally or in CI, before merge:

- `poetry run pytest` with coverage of `app` (branch coverage in CI, report
  uploaded to Codecov).
- `poetry run ruff check .` with the repository Ruff configuration.
- `poetry run mypy .` with the repository MyPy configuration
  (`disallow_untyped_defs` and related strict flags).

CI is `.github/workflows/ci.yaml` and MUST stay green on the pull request.
Plans produced under Speckit MUST include a Constitution Check against this
document. `/speckit.analyze` MUST treat a conflict with a MUST principle as
CRITICAL.

## Governance

This constitution supersedes informal practice for this repository. Speckit
templates and scripts under `.specify/` define the workflow; they do not
override these principles.

Amendments MUST be made in `.specify/memory/constitution.md`, reviewed in a
pull request, and versioned with SemVer:

- MAJOR: a principle is removed or redefined.
- MINOR: a principle or section is added.
- PATCH: wording is clarified without changing the obligation.

`Ratified` stays at the date of the first accepted version. `Last Amended`
updates on every amendment. Compliance is verified in review and in the
Constitution Check of each plan. A conflict with a MUST principle is CRITICAL
and MUST be resolved or explicitly amended before implementation. Complexity
that exceeds these principles MUST be justified in the engineering spec.

**Version**: 1.0.0 | **Ratified**: 2026-10-01 | **Last Amended**: 2026-10-01
