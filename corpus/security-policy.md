# Meridian Information Security Policy

This policy defines the mandatory security controls for all Meridian systems and
data. It applies to every employee, contractor, and third party with access to
Meridian systems. Violations may result in disciplinary action.

## Access Control

Access to systems and data follows the principle of least privilege: you receive
only the access required for your role. Access is reviewed quarterly by system
owners, and access that has not been used for 90 days is automatically revoked.

Requests for new access must be approved by your manager and the owner of the
target system through the access request workflow in the internal portal.

## Authentication and Multi-Factor Authentication

Multi-factor authentication (MFA) is mandatory for all Meridian accounts,
including email, the code repository, cloud consoles, and the VPN. Hardware
security keys (FIDO2) are the required second factor for engineers with
production access. TOTP authenticator apps are acceptable for all other staff.
SMS-based codes are not permitted as a second factor.

## Password Requirements

Passwords must be at least 14 characters. Meridian issues every employee a
password manager licence, and reusing passwords across services is prohibited.
Passwords are not rotated on a fixed schedule; instead, they are changed
immediately if a compromise is suspected. Never share a password, and never
enter a Meridian password into a site reached through a link in an email.

## Data Classification

Meridian data is classified into four tiers:

- **Public**: may be shared freely (e.g. marketing material).
- **Internal**: default for business data; shareable within Meridian only.
- **Confidential**: sensitive business or personal data; access on a need-to-know
  basis and encrypted in transit and at rest.
- **Restricted**: the most sensitive data, including customer financial data and
  secrets. Access requires explicit sign-off from the data owner and the
  Security team, and all access is logged.

Customer personal data is always at least Confidential. Production credentials,
API keys, and encryption keys are always Restricted.

## Device Security

Company laptops must have full-disk encryption enabled and the managed endpoint
agent installed. The screen must lock automatically after five minutes of
inactivity. Personal devices may access email and chat only through the approved
mobile applications, which enforce a separate work profile.

## Reporting a Security Incident

If you suspect a security incident — a lost device, a phishing click, exposed
credentials, or unusual account activity — report it immediately to the Security
team via the #security-incidents channel or security@meridian.example. Do not
attempt to investigate or remediate on your own. Early reporting is never
penalised, even if you caused the incident. For the response process itself, see
the Incident Response Runbook.
