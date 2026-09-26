# Security reporting

Report a suspected vulnerability privately to **ujjwalsrao@gmail.com** with
`pygarble security` in the subject. Do not post active credentials, personal
data or exploit details in a public issue. Use synthetic or redacted examples.

Include the package version, Python version, enabled categories/backends,
configuration, impact and a minimal reproduction. Maintainers review reports
on a best-effort basis; there is no guaranteed response time or support SLA.

Security fixes are targeted at the current development branch and the next
release. There is no separate long-term-support or backport commitment.

pygarble is a heuristic first pass. A clean scan is not proof that text contains
no sensitive data or abuse. Backend errors mean the scan is incomplete.
Ordinary false positives and missing detection rules can be reported as bugs
with synthetic inputs. See the documentation for known coverage limits.
