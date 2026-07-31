# Engineering Onboarding Guide

Welcome to the Meridian engineering team. This guide gets you from a fresh laptop
to shipping your first change. It complements, and does not override, the
Information Security Policy.

## Development Environment

Meridian services are written primarily in Python (FastAPI) and TypeScript. We
target Python 3.11. Set up your environment by cloning the `platform` monorepo
and running `make bootstrap`, which installs dependencies, provisions local
Postgres via Docker, and runs the test suite to confirm a working setup.

You will be issued access to the code repository, the cloud console, and the CI
system on your first day. All three require MFA with a hardware key, which the IT
team provisions during onboarding.

## Git Workflow

We use trunk-based development. Create a short-lived feature branch off `main`,
open a pull request early, and keep changes small. Every pull request requires:

- At least one approving review from a code owner.
- A green CI run (lint, type-check, unit tests, and integration tests).
- A linked issue or ticket describing the change.

Direct pushes to `main` are blocked. Commits must be signed. We squash-merge, so
your branch history does not need to be tidy, but the final PR title becomes the
commit message and should follow Conventional Commits.

## Testing Expectations

New code must ship with tests. We aim for meaningful coverage of behaviour, not a
coverage percentage target. Unit tests should not hit the network; use the
provided fakes for external services. Integration tests run against ephemeral
Postgres and are allowed to be slower.

## Deployments

Deployments are continuous. Merging to `main` triggers a pipeline that builds a
container, runs the full test suite, deploys to staging, runs smoke tests, and —
if those pass — promotes to production automatically. There is no manual release
step for standard changes.

Feature flags gate risky changes. Wrap new behaviour in a flag, deploy it off,
and enable it gradually. Roll back by disabling the flag rather than reverting
the deploy where possible.

## On-Call

Every engineer joins the on-call rotation after their first month. On-call weeks
run Monday to Monday. The primary on-call engineer acknowledges pages within 15
minutes during working hours and within 30 minutes out of hours. If you cannot
resolve an incident, escalate to the secondary on-call and, for customer-facing
outages, follow the Incident Response Runbook. On-call is compensated with a
weekly stipend and time off in lieu for out-of-hours incidents.

## Getting Help

Ask questions early in your team channel — nobody expects you to know the codebase
in week one. Each new engineer is assigned an onboarding buddy for their first
month who is your first port of call for anything, technical or otherwise.
