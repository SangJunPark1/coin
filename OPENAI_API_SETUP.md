# OpenAI API Setup

This project can use OpenAI as an AI decision reviewer for paper-trading candidates.

## Secret Name

Use this exact GitHub repository secret name:

```text
OPENAI_API_KEY
```

Do not put the API key in source files, `config.cloud.json`, logs, or reports.

## GitHub Setup

1. Open the GitHub repository.
2. Go to `Settings`.
3. Open `Secrets and variables`.
4. Open `Actions`.
5. Click `New repository secret`.
6. Set `Name` to `OPENAI_API_KEY`.
7. Paste the OpenAI API key into `Secret`.
8. Click `Add secret`.

The workflows pass this secret into the simulation step as an environment variable.

## Runtime Behavior

The cloud config uses:

```json
"ai_decision": {
  "enabled": true,
  "provider": "openai",
  "min_confidence": 0.55,
  "openai_model": "gpt-5.4-mini",
  "api_key_env": "OPENAI_API_KEY"
}
```

If the secret is present and the OpenAI request succeeds, the report will show the AI decision source as `openai`.

If the key is missing, the model is unavailable, or the request fails, the system falls back to the local reviewer and marks the source as `local-fallback:openai`. Trading does not stop just because the AI API is unavailable.

## Safety Rules

- OpenAI only reviews candidate decisions.
- The deterministic risk manager still controls position limits, loss limits, stops, cooldowns, and forced exits.
- Real orders are not implemented in this paper-trading system.

