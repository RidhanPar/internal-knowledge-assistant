# Incident Response Runbook

This runbook defines how Meridian responds to production incidents — customer-
facing outages, degradations, and security events. It is the operational
companion to the Information Security Policy.

## Severity Levels

Incidents are classified by severity, which sets the response expectations:

- **SEV1**: Critical. Complete outage or data loss affecting many customers, or a
  confirmed security breach. Requires immediate, all-hands response.
- **SEV2**: Major. Significant degradation or a feature down for many customers,
  with no full workaround.
- **SEV3**: Minor. Limited impact, a workaround exists, or a single customer is
  affected.

When in doubt, declare the higher severity. Severity can be downgraded later.

## Declaring an Incident

Anyone can declare an incident. To declare, post in the #incidents channel using
the `/incident` command, which creates a dedicated incident channel and pages the
on-call engineer. Declaring early is always the right call — an incident that
turns out to be minor costs little, but a delayed response to a real outage is
expensive.

## Roles During an Incident

- **Incident Commander (IC)**: coordinates the response, makes decisions, and is
  the single point of accountability. The IC does not fix the problem themselves.
- **Operations Lead**: performs the technical investigation and remediation.
- **Communications Lead**: writes customer and internal updates.

For a SEV1, the IC must be someone trained in the role; the on-call engineer
holds IC until a trained IC takes over. For SEV2 and SEV3, the on-call engineer
may hold all roles if the incident is small.

## Communication Cadence

For SEV1 incidents, post an internal status update at least every 30 minutes,
even if the update is "no change". Customer-facing status page updates are posted
by the Communications Lead and must be approved by the IC. Never speculate about
root cause in customer communications before it is confirmed.

## Resolution and Postmortem

An incident is resolved when customer impact has ended and the fix is verified.
Every SEV1 and SEV2 incident requires a blameless postmortem within five working
days. The postmortem documents the timeline, contributing factors, and action
items, and focuses on systemic causes rather than individual blame. Action items
are tracked to completion like any other work.

## Security Incidents

A suspected security breach is at least a SEV2 and is escalated to the Security
team immediately, in parallel with the standard response. Preserve evidence: do
not power off affected machines or delete logs. The Security team decides on
containment steps, legal notification, and any customer disclosure.
