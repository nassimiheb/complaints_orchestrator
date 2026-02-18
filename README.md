# Customer Complaints Resolution Orchestrator

Phase 0 bootstrap for a multi-agent complaint resolution workflow.

## Quick Start
1. Create a virtual environment and activate it.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy env template and update values as needed:
   - `copy .env.example .env`
4. Validate bootstrap:
   - `python src/main.py --help`

## Current Scope
This bootstrap includes:
- Repository structure under `src/complaints_orchestrator/`
- Base config loading from environment variables
- Base logging wiring
- CLI entrypoint scaffold
