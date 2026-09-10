# server/tests/

Fast, API-free pytest unit suite — pure logic only (chunking, expression
evaluation, response parsing, the malformed-response/citation guards, tool
dispatch error handling). No network calls to Gemini, Qdrant, or Supabase;
safe to run offline or in CI.

Run with:

```
pytest server/tests/ -v
```

This is a different thing from `server/scripts/verify_*.py`, which are
manual smoke-test scripts that make real calls to the live services
(Gemini/Qdrant/Supabase) and are meant to be run by hand against a
configured `.env`, not as an automated test suite. Keep new pure-logic unit
tests here as `test_*.py`; keep new live-service smoke tests as
`scripts/verify_*.py`.
