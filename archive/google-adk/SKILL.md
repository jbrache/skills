---
name: google-adk
description: "Build AI agents, chatbots, and autonomous assistants with tools, memory, and multi-agent systems using Python. Use when: (1) Creating agents/chatbots/assistants that use tools (Google Search, APIs, custom functions, databases), (2) Building multi-agent systems with specialized agents that collaborate, (3) Agents that remember conversation history or maintain state, (4) Deploying agents to Cloud Run, Vertex AI Agent Engine, or GKE, (5) Evaluating agent performance with test sets, (6) User mentions 'agent', 'chatbot', 'assistant', 'autonomous', 'multi-agent', 'ADK', or 'agentic'. Choose this over vertex-ai skill when building complete agents with tools and orchestration, not just making model API calls."
---

# Google Agent Development Kit (ADK)

Open-source, code-first Python framework for building, evaluating, and deploying AI agents.

## Key Features

- **Code-first development**: Define agents, tools, and orchestration in Python
- **Rich tool ecosystem**: Google Search, OpenAPI, MCP, custom functions, Google Cloud tools
- **Multi-agent systems**: Compose specialized agents into hierarchies
- **Model-agnostic**: Optimized for Gemini, compatible with other models
- **Deploy anywhere**: Cloud Run, Vertex AI Agent Engine, GKE

## Prerequisites

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Set your project
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

**Quick Vertex AI setup:**
```bash
echo "GOOGLE_GENAI_USE_VERTEXAI=true" >> .env
echo "GOOGLE_CLOUD_LOCATION=global" >> .env
```

See [Configuration](references/configuration.md) for all auth methods.

## Installation

```bash
pip install google-adk
# or
uv add google-adk
```

## Quick Start

### 1. Create Project

```bash
mkdir -p weather/weather-agent && cd weather
uv init && uv add google-adk
```

### 2. Create Your Agent

```python
# weather-agent/agent.py
from google.adk.agents import LlmAgent

def get_weather(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: City name

    Returns:
        Weather information
    """
    return f"Weather in {location}: Sunny, 22°C"

# ADK discovers this 'root_agent' variable automatically
root_agent = LlmAgent(
    name="weather_assistant",
    model="gemini-3-flash-preview",
    instruction="You are a helpful weather assistant. Use the get_weather tool to answer questions.",
    description="Provides weather information",
    tools=[get_weather]
)
```

### 3. Test Your Agent

```bash
# Interactive UI
uv run adk web
# Opens http://localhost:8080 - select your agent and chat

# Or use replay file
uv run adk run weather-agent/ --replay test.json
```

### 4. Deploy (Optional)

```bash
# Deploy to Vertex AI Agent Engine (recommended)
uv run adk deploy weather-agent/ --platform vertex-agent-engine

# Or deploy to Cloud Run
uv run adk deploy weather-agent/ --platform cloud-run
```

See [Deployment](references/deployment.md) for production patterns.

---

## Quick Reference

### Multi-Turn Conversation

```python
from google.adk.sessions import Session

session = Session(session_id="user-123")
response1 = agent.run(input="My name is Alice", session=session)
response2 = agent.run(input="What's my name?", session=session)  # Remembers context
```

### Structured Output

```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    location: str
    temperature: float
    conditions: str

agent = LlmAgent(
    name="weather_agent",
    model="gemini-3-flash-preview",
    instruction="Return weather data",
    response_model=WeatherReport  # Enforces JSON schema
)
```

### Testing with Replay File

```json
{
  "state": {"session_id": "test-001", "contents": []},
  "queries": ["What's the weather in Tokyo?"]
}
```

```bash
uv run adk run weather_agent/ --replay test.json
```

### Model Selection

| Model | Use For |
|-------|---------|
| `gemini-3-flash-preview` | Most tasks (default) |
| `gemini-3-pro-preview` | Complex reasoning (requires `GOOGLE_CLOUD_LOCATION=global`) |
| `gemini-2.5-flash-lite` | High volume, cost-sensitive |

## Related Skills

| Skill | When to Use |
|-------|-------------|
| **[vertex-ai](../vertex-ai/SKILL.md)** | Direct model API calls without agent framework |
| **[a2a](../a2a/SKILL.md)** | Agent-to-agent communication across distributed services |
| **[vertex-agent-engine](../vertex-agent-engine/SKILL.md)** | Deploy agents to managed infrastructure |

## Reference Documentation

- **[agents.md](references/agents.md)** - Agent types, multi-agent systems, workflow patterns, instruction engineering
- **[tools.md](references/tools.md)** - Built-in tools, custom tools, OpenAPI, MCP, Google Cloud integrations
- **[configuration.md](references/configuration.md)** - Models, sessions, callbacks, runtime config, safety, auth
- **[development.md](references/development.md)** - Project structure, testing, replay files, common errors
- **[deployment.md](references/deployment.md)** - Vertex AI Agent Engine, Cloud Run deployment
- **[error-handling.md](references/error-handling.md)** - Exception handling, retry strategies

## Additional Resources

- **Official Documentation**: https://github.com/google/adk-docs
- **Repository**: https://github.com/google/adk-python
- **API Reference**: https://github.com/google/adk-docs/blob/main/docs/api-reference/python/
