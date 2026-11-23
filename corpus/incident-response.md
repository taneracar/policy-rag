---
id: incident-response
title: Production Incident Response Policy
effective: 2026-04-01
status: active
owner: Engineering
---
# Production Incident Response Policy

## Severity levels
SEV1 is a full outage or data loss affecting customers. SEV2 is a major feature broken for many customers. SEV3 is a minor issue with a workaround.

## Response times
The on-call engineer acknowledges a SEV1 page within 15 minutes and a SEV2 page within 30 minutes. SEV3 issues are handled during business hours. If the primary on-call does not acknowledge in time, the page escalates to the secondary on-call and then to the engineering manager.

## Communication
For SEV1 and SEV2 incidents, an incident commander is named and a status page update is posted within 30 minutes of declaring the incident, then at least every 60 minutes until it is resolved.

## On-call
Engineers join the on-call rotation after 90 days of employment. A rotation shift lasts one week. On-call engineers receive a stipend of $400 per week of on-call and may take one comp day after a week with a SEV1 page outside business hours.

## Postmortems
Every SEV1 and SEV2 incident has a blameless postmortem. The written postmortem is due within 5 business days of resolution and is reviewed at the weekly operations review.
