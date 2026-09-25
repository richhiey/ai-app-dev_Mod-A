# HelioDesk FieldCare Sprint 4 Assets

This folder contains the single synthetic FieldCare environment used across the whole Sprint 4 project. The same documents, equipment records, maintenance history, tickets, tools, requests, and evaluation cases support the four Campus Independent Practice lessons, LS13, LS14, and the shared Colab notebook.

The records describe a fictional course snapshot dated `2026-09-09`. Tools query
these records directly; they do not contact an external equipment service.
Model generation, embeddings, and reranking use real OpenRouter calls.

## Asset map

| File | What it contains | Application component |
|---|---|---|
| `fieldcare_manifest.json` | Canonical asset index, usage map, and consistency rules | Course and project planning |
| `service_docs.jsonl` | Troubleshooting, safety, warranty, model, parts, escalation, and legacy documents | Retrieval and reranking |
| `equipment_records.csv` | Structured equipment, site, customer, warranty label, and configuration records | `get_equipment_record` |
| `maintenance_history.csv` | Prior visits, replaced parts, recurring faults, deferred work, and unresolved state | `get_maintenance_history` |
| `service_tickets.csv` | Current ticket status, priority, escalation state, and next action | `get_ticket_status` |
| `tool_schemas.json` | MCP-style callable contracts and safe-use rules | Tool planning and orchestration |
| `user_requests.jsonl` | Technician request bank with expected source needs | Scoping and manual tests |
| `eval_cases.jsonl` | Expected behavior test suite | Debugging map and edge-case improvement |

## Teaching design

The corpus intentionally includes:

- strongly relevant HX overheating, filter, airflow, sensor, warranty, and escalation documents;
- similar but less relevant VX documentation;
- overlapping terms such as `overheating`, `filter`, `airflow`, `E-117`, and `sensor`;
- a legacy bulletin, `DOC-FC-LEG-2019`, that looks useful but is superseded;
- cases where equipment, maintenance history, warranty status, and current ticket state change the response;
- evaluation scenarios for unavailable warranty data and timeouts. These describe
  expected behavior, not guaranteed outcomes: the application never injects a
  failure for a particular equipment ID. Unit tests cover failure handling with
  isolated test doubles; the live notebook records what actually happened.

## Cross-reference rules

- Use `equipment_id` and `ticket_id` as unique identifiers in learner-facing explanations.
- Do not answer warranty questions from documentation alone. Use `get_warranty_status`.
- Do not infer current escalation state from old maintenance notes. Use `get_ticket_status`.
- Use `DOC-FC-SAF-001` as an override when a request contains safety indicators.
- Use `DOC-FC-ESC-007` plus maintenance history for recurring fault escalation.
- Treat `DOC-FC-LEG-2019` as intentionally outdated evidence. It may be retrieved, but it should not drive the final answer.

## Sprint usage

| Sprint component | Main assets used |
|---|---|
| Campus 1 | Manifest, request bank, data-source map |
| Campus 2 | Tool schemas, evaluation cases, response flags |
| Campus 3 | Full data package and shared Colab |
| Campus 4 | Evaluation cases and observed traces |
| LS13 | Scope examples, source/tool split, checkpoint answer guide |
| LS14 | Integrated path, edge cases, observed traces |
| Shared Colab | All files in this folder |
