# Customer Complaints Resolution Orchestrator

Multi-agent LangGraph workflow for omnichannel fashion retail complaints (FR/EN), using Mistral for reasoning, Chroma RAG for policy grounding, SQLite for memory, and local mock tools.

## What the system produces
For each complaint run, the workflow outputs:
1. Structured case analysis (type, sentiment, urgency, language, risk flags)
2. Resolution decision (`REFUND`, `VOUCHER`, `EXCHANGE`, `ESCALATE`, `INFO_ONLY`)
3. Ready-to-send customer email in FR or EN
4. Executed tool actions and memory updates
5. Security output (`security_events`, `output_guard_passed`)

## Architecture at a glance
### Agents
1. Triage and Routing Agent
- Classifies complaint and risk flags
- Detects language and response language
- Chooses route (`ESCALATE_IMMEDIATE` or `NEED_CONTEXT`)

2. Context and Policy Agent
- Calls read tools:
  - `get_customer_profile`
  - `get_order_details`
  - `get_case_history`
- Retrieves policy/tone constraints from Chroma
- Minimizes tool payloads before model reasoning

3. Resolution Strategist and Email Writer Agent
- Scores options and selects decision
- Enforces HITL rules
- Calls action tools:
  - `create_compensation`
  - `issue_refund`
  - `create_support_ticket`
- Applies output guard before final email

### Orchestration (LangGraph)
Implemented nodes:
- `ingest_email_node`
- `triage_router_node`
- `context_policy_node`
- `resolution_node`
- `finalize_node`

Routing rules:
- Legal/public risk in triage -> `ESCALATE_IMMEDIATE` path
- Otherwise -> context retrieval before resolution
- Finalize persists structured memory only

## Project layout
```text
complaints_orchestrator/
  data/                         # Mock customers, orders, historical cases, sample scenario inputs
  docs/                         # Architecture and presentation docs
  eval/
    run_scenarios.py            # Evaluation runner
    scenarios.json              # Evaluation scenarios and expected outcomes
  scripts/                      # Agent playgrounds
  src/
    main.py                     # End-to-end CLI entrypoint
    complaints_orchestrator/
      agents/
      tools/
      rag/
      memory/
      utils/
      constants.py
      config.py
      graph.py                  # LangGraph orchestration
      logging_config.py
      state.py                  # Typed CaseState contract
  tests/                        # Unit tests
  .env.example
  requirements.txt
```

## Prerequisites
- Python 3.11+
- Mistral API key

## Setup
From `complaints_orchestrator/`:

1. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create `.env`
```bash
cp .env.example .env
```

4. Set required variables in `.env`
- `MISTRAL_API_KEY`
- Optional tuning:
  - `CCO_MODEL_NAME`
  - `CCO_CHROMA_DIR`
  - `CCO_SQLITE_PATH`
  - `CCO_HITL_AMOUNT_THRESHOLD`
  - `CCO_LOW_CONFIDENCE_THRESHOLD`

## Build local artifacts 
Set `PYTHONPATH` so package imports resolve from `src/`.

```bash
export PYTHONPATH=src
```

1. Build RAG index
```bash
python src/complaints_orchestrator/rag/build_index.py
```

2. Seed persistent memory
```bash
python src/complaints_orchestrator/memory/seed_memory.py --db-path ./storage/complaints_memory.db
```

## Run end-to-end workflow
Main CLI:
```bash
python src/main.py --help
```

Run a scenario by id:
```bash
python src/main.py --scenario 1
```

Use a custom scenario file:
```bash
python src/main.py --scenario 1 --scenarios-file ./data/triage_playground_cases.json
```

Skip index rebuild for faster reruns:
```bash
python src/main.py --scenario 1 --skip-index-build
```

Security utility demo:
```bash
python src/main.py --demo-security
```

## Runtime output contract (main CLI)
Each run prints:
- Case summary (type, sentiment, urgency, language)
- Retrieved policy sources (doc IDs)
- Decision and rationale
- Tool actions taken
- Final email subject and body
- Security output (`security_events`, `output_guard_passed`)

## Evaluation harness
Scenario file:
- `eval/scenarios.json`

Run all eval scenarios :
```bash
python eval/run_scenarios.py
```

Run one scenario:
```bash
python eval/run_scenarios.py --scenario-id eval_refund_defective_en --skip-index-build
```

Summary only:
```bash
python eval/run_scenarios.py --quiet
```

Evaluation asserts:
- Expected route/language/decision behavior
- No internal leakage in final email
- No raw email persistence to SQLite
- No redaction placeholder leakage in customer-facing email

## Agent playgrounds (isolated testing)
Use these to debug one agent at a time:

1. Triage only
```bash
python scripts/triage_playground.py --list
python scripts/triage_playground.py --case 1
```

2. Context + policy only (triage stubbed)
```bash
python scripts/context_policy_playground.py --case 1
```

3. Resolution only (triage/context stubbed)
```bash
python scripts/resolution_playground.py --case 1
```

Important:
- Playground scripts are for isolated behavior checks.
- Persistent memory write path is in graph `finalize_node`, so playground runs are not full workflow equivalents.

## Security model
Implemented controls:
- PII redaction before triage model call
- Strict schema validation at tool boundaries
- Role-based tool permissions:
  - read tools for `context_policy_node`
  - action tools for `resolution_node` only
- RAG hardening:
  - internal source allowlist
  - prompt-injection checks at indexing and retrieval
  - chunk/excerpt capping and directive stripping
- Output guard before customer response:
  - blocks internal scores, policy IDs, raw tool JSON, raw RAG metadata
  - sanitize/fallback handling
- No raw email persistence in memory adapter

## Memory behavior
SQLite DB (default): `./storage/complaints_memory.db`

Tables:
- `customers_memory`
- `cases_memory`

Writes happen in `finalize_node` through `record_finalize_update(...)`.

Quick check:
```bash
python -c "import sqlite3; c=sqlite3.connect('storage/complaints_memory.db'); print(c.execute('SELECT case_id, decision, status, compensation_value FROM cases_memory ORDER BY updated_at DESC LIMIT 5').fetchall()); c.close()"
```

## Tests
Run all tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
