# Error Handling

Essential patterns for handling errors in ADK agents.

## Basic Pattern

Always wrap agent execution in try/except:

```python
from google.adk.runtime import RunError
from google.api_core import exceptions

def run_agent_safely(agent, user_input):
    try:
        response = agent.run(input=user_input)
        return response.text

    except exceptions.ResourceExhausted:
        return "Service at capacity. Please try again."

    except exceptions.InvalidArgument as e:
        return f"Invalid request: {e}"

    except RunError as e:
        return f"Agent error: {e}"

    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred."
```

## Retry Pattern

Implement exponential backoff for transient errors:

```python
import time
from google.api_core import exceptions

def run_with_retry(agent, user_input, max_retries=3):
    for attempt in range(max_retries):
        try:
            return agent.run(input=user_input)

        except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable):
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(wait_time)

        except exceptions.DeadlineExceeded:
            raise  # Don't retry timeouts

    raise RuntimeError("Max retries exceeded")
```

Or use RunConfig for automatic retries:

```python
from google.adk.runtime import RunConfig

config = RunConfig(
    retry_config={
        "max_retries": 3,
        "backoff_factor": 2,
        "retry_on": ["ResourceExhausted", "ServiceUnavailable"]
    }
)
```

## Common Errors Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `ResourceExhausted` | API quota exceeded | Wait and retry, or request quota increase |
| `InvalidArgument` | Invalid model name or parameters | Check model name, verify config |
| `PermissionDenied` | Missing credentials | Run `gcloud auth application-default login` |
| `DeadlineExceeded` | Request timeout | Increase timeout or simplify request |
| `ServiceUnavailable` | Temporary outage | Retry with exponential backoff |

## ADK-Specific Errors

### Directory Naming Error

```
pydantic_core._pydantic_core.ValidationError: Invalid app name 'my-agent'
```

**Fix:** Use underscores, not hyphens: `my_agent/`

### Missing State Field

```
pydantic_core._pydantic_core.ValidationError: state - Field required
```

**Fix:** Replay files need `state` wrapper:
```json
{"state": {"session_id": "test", "contents": []}, "queries": [...]}
```

### Agent Not Found

```
No agent found in specified directory
```

**Fix:**
- File must be `agent.py`
- Variable must be `root_agent`
- Must be at module level

### Vertex AI Configuration

```
ValueError: Missing key inputs argument!
```

**Fix:** Add to `.env`:
```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_LOCATION=global
```

### Model Location Mismatch

```
Model gemini-3-pro-preview not available in us-central1
```

**Fix:** Use `GOOGLE_CLOUD_LOCATION=global` for preview models.

### Streaming Incompatibility

```
Streaming not supported with function calling
```

**Fix:** Disable streaming when using tools or `response_model`.

## Debugging

Enable verbose logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("google.adk")
logger.setLevel(logging.DEBUG)
```

## Documentation References

- Google Cloud Errors: https://cloud.google.com/apis/design/errors
- ADK Runtime: https://github.com/google/adk-docs/blob/main/docs/runtime/index.md
