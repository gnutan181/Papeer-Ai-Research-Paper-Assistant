import os

# The suite never calls live APIs. Guardrails are switched off for the
# existing tests (they then behave exactly as before guardrails existed);
# tests/test_guardrails.py enables them explicitly with a mocked Groq client.
os.environ.setdefault("GUARDRAILS_ENABLED", "false")
