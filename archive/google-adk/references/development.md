# Development Guide

Project structure, local development, and testing for ADK agents.

## Project Structure

ADK uses a workspace pattern where each agent is a subdirectory:

```
my-workspace/
├── weather_agent/
│   ├── agent.py          # Entry point with root_agent variable
│   ├── tools.py          # Custom tools (optional)
│   ├── requirements.txt
│   └── .env              # Local dev secrets (never commit)
└── search_agent/
    ├── agent.py
    └── requirements.txt
```

### Critical Requirements

**Directory naming**: Use snake_case only (letters, digits, underscores).

```bash
# Wrong - causes validation error
bartender-agent/    # Hyphens not allowed

# Correct
bartender_agent/    # Snake_case
```

**Entry point**: File must be `agent.py` with `root_agent` variable at module level.

```python
# agent.py
from google.adk.agents import LlmAgent

root_agent = LlmAgent(  # Must be named 'root_agent'
    name="my_assistant",
    model="gemini-3-flash-preview",
    instruction="You are a helpful assistant.",
    tools=[...]
)
```

### Tool Organization

**1-5 tools**: Define in agent.py directly.

**6+ tools**: Separate file:
```python
# tools.py
from google.adk.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description."""
    return result
```

```python
# agent.py
from tools import my_tool
root_agent = LlmAgent(tools=[my_tool])
```

---

## Local Development

### Web Interface

```bash
# Launch interactive UI
uv run adk web
# Opens http://localhost:8080
```

### Replay File Testing

Test with reproducible JSON input:

```bash
uv run adk run weather_agent/ --replay test.json
```

**Replay file format** (required structure):

```json
{
  "state": {
    "session_id": "test-001",
    "contents": []
  },
  "queries": [
    "What's the weather in Tokyo?",
    "How about tomorrow?"
  ]
}
```

### Multi-Turn Conversation Test

```json
{
  "state": {
    "session_id": "context-test",
    "contents": []
  },
  "queries": [
    "My name is Alice and I live in Seattle",
    "What's the weather in my city?",
    "What's my name?"
  ]
}
```

---

## Evaluation Sets

For systematic testing:

```bash
# Run evaluation
uv run adk eval weather_agent/ eval_set.evalset.json

# Evaluation set format
```

```json
{
  "version": "1.0",
  "test_cases": [
    {
      "id": "test_001",
      "input": "What's the weather in Paris?",
      "expected_tools": ["get_weather"],
      "expected_keywords": ["Paris", "temperature"]
    }
  ]
}
```

---

## Common Errors

### Directory Naming Error

```
pydantic_core._pydantic_core.ValidationError: Invalid app name 'my-agent'
```

**Fix:** Use underscores: `my_agent/`

### Missing State Field

```
pydantic_core._pydantic_core.ValidationError: state - Field required
```

**Fix:** Wrap session_id in state object:
```json
{"state": {"session_id": "...", "contents": []}, "queries": [...]}
```

### Agent Not Found

```
No agent found in specified directory
```

**Fix:**
- File must be `agent.py` (not main.py)
- Variable must be `root_agent` (not agent)
- Must be at module level (not inside function)

### Model Location Mismatch

```
Model gemini-3-pro-preview not available in us-central1
```

**Fix:** Use `GOOGLE_CLOUD_LOCATION=global` for preview models.

### Vertex AI Configuration

```
ValueError: Missing key inputs argument!
```

**Fix:** Add to `.env`:
```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_LOCATION=global
```

---

## Testing Best Practices

1. **Test tools independently** before giving to agents
2. **Use descriptive session IDs** (`test-weather-multi-city-001`)
3. **Version control test files** for reproducibility
4. **Test error cases** not just happy paths

### Test File Organization

```
my-workspace/
├── weather_agent/
│   ├── agent.py
│   └── tests/
│       ├── basic.json
│       ├── errors.json
│       └── multi_turn.json
```

---

## Deployment Prep

When ready to deploy, add serve() to agent.py:

```python
from google.adk.runtime import serve
import os

root_agent = LlmAgent(...)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    serve(root_agent, host="0.0.0.0", port=port)
```

See [deployment.md](deployment.md) for Cloud Run and Vertex AI deployment.

## Documentation References

- Project structure: https://github.com/google/adk-docs/blob/main/docs/get-started/project-structure.md
- Testing: https://github.com/google/adk-docs/blob/main/docs/evaluate/index.md
