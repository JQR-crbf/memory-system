# Public Sanitization Notes

This directory is generated from a local private memory-system instance.
It contains code and integration setup, but runtime personal data is removed.

Sanitized items:
- Removed `.env` and replaced `.env.example` with placeholders
- Replaced `data/L0/profile.yaml` and `data/L5/goals.yaml` with templates
- Cleared system logs under `data/system/*.jsonl`
- Replaced classifier learned rules with empty defaults
- Removed vector/graph runtime storage (`data/qdrant`, `data/neo4j`) and kept `.gitkeep`

To regenerate: run `python scripts/export_public_sanitized.py --force` in source repo.
