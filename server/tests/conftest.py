"""Shared pytest setup for server/tests/ -- the fast, API-free unit suite.

Distinction from server/scripts/verify_*.py: those scripts make real calls to
Gemini/Qdrant/Supabase and are run manually as smoke tests against live
services. Everything under server/tests/ is a pytest unit test that exercises
pure logic only (chunking, expression evaluation, response parsing, the
malformed-response/citation guards, tool-dispatch error handling) with no
network I/O -- safe to run in CI or offline, and fast.

Importing app.* modules transitively imports app.config, which -- like every
other script in this project (see run_eval.py's docstring) -- resolves its
".env" path relative to the current working directory, and server/.env is
where the real values live. So this chdir's into server/ before any test
module imports app.*, regardless of what directory pytest was invoked from
(e.g. `pytest server/tests/ -v` from the repo root). This only lets
app.config's Settings() validation succeed at import time -- constructing the
Gemini/Qdrant/Supabase client objects imported transitively by agent_loop.py
does not itself make any network call, so no test here actually talks to a
live service.
"""

import os
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))
os.chdir(SERVER_DIR)
