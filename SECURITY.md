# Security policy

GPU Steward launches user-provided commands and controls access to expensive
accelerators. Please do not disclose a suspected vulnerability in a public
issue before it has been assessed.

Report vulnerabilities through the repository's GitHub private vulnerability
reporting / Security Advisory interface. Include the affected version, a
minimal reproduction, the expected impact, and any known mitigation. Do not
include credentials, SSH keys, private hostnames, or production data.

The project never needs SSH private keys. It relies on the caller's existing
SSH configuration and must not terminate GPU processes that it does not own.
