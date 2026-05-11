# Security Policy

## Reporting

Report vulnerabilities responsibly by email to **g@abejar.net** — do not open
public issues for security-sensitive findings.

## Scope

Server-side telemetry analytics for query streams. The detector inspects only
metadata and query/response text that the operator has already chosen to
log; it does not perform any active probing of clients.

## Considerations

- The optional LLM analyst transmits trigger summaries and the numeric
  feature dictionary (no raw queries by default) to the configured LLM
  endpoint. Review your data-handling requirements before enabling it.
- Rules are intentionally **high-recall** to catch slow, low-volume
  distillation; you should tune thresholds for your traffic profile before
  using them for automated blocking.
- The IsolationForest baseline is optional and requires the operator to
  supply a labelled benign corpus; pickled `*.joblib` artifacts must be
  treated as code (the standard sklearn warning applies).

## Contact

Responsible disclosure: **g@abejar.net**
