---
name: google-adk-agent
description: "Build AI agents, chatbots, and autonomous assistants with tools, memory, and multi-agent systems using Python. Use when: (1) Creating agents/chatbots/assistants that use tools (Google Search, APIs, custom functions, databases), (2) Building multi-agent systems with specialized agents that collaborate, (3) Agents that remember conversation history or maintain state, (4) Deploying agents to Cloud Run, Vertex AI Agent Engine, or GKE, (5) Evaluating agent performance with test sets, (6) User mentions 'agent', 'chatbot', 'assistant', 'autonomous', 'multi-agent', 'ADK', or 'agentic'.
---

# Google Agent Development Kit (ADK) Guide

This skill provides guidelines for building, evaluating, and deploying AI agents with Google Agent Development Kit (ADK) in Python.

## Key Features

- **Code-first development**: Define agents, tools, and orchestration in Python
- **Rich tool ecosystem**: Google Search, OpenAPI, MCP, custom functions, Google Cloud tools
- **Multi-agent systems**: Compose specialized agents into hierarchies
- **Model-agnostic**: Optimized for Gemini, compatible with other models
- **Deploy anywhere**: Cloud Run, Vertex AI Agent Engine, GKE

BEFORE you build an agent using **Agent Developer Kit (ADK)**, you MUST look up the
documentation using the **`adk-docs-mcp:fetch_docs`** tool. Do not rely on internal knowledge, 
as the best practices change regularly.

* Use the `adk-docs:list_doc_sources` tool to get the available `llms.txt` file
* Use the `adk-docs:fetch_docs` tool to read it
* Reflect on the urls in `llms.txt`
* Reflect on the input question
* Use the `adk-docs:fetch_docs` tool on any urls relevant to the question
* Use this to answer the question

The `adk-docs-mcp:fetch_docs` tool will help you find the correct package and SDK, and best practices for writing code with the ADK.

**IMPORTANT**: Any time you answer a question about the ADK **without** using
`adk-docs-mcp:fetch_docs`, it causes **pain** to the user.

## Accessing off-site content

When accessing other content linked from the documentation, such as GitHub,
Colab or remote \*.ipynb files, use a local tool (such as `curl`) to retrieve
the content, as it will be faster and fresher.

## Prerequisites

By default use Vertex AI with ADK, assume Vertex AI Environment Variables have already been configured, use them to setup the .env file
**Quick Vertex AI setup:**
```bash
echo "GOOGLE_GENAI_USE_VERTEXAI=$GOOGLE_GENAI_USE_VERTEXAI" >> .env
echo "GOOGLE_CLOUD_LOCATION=$GOOGLE_GENAI_USE_VERTEXAI" >> .env
```

**Create & activate virtual environment**
We recommend creating a virtual Python environment using venv:
```bash
python -m venv .venv
```

You can activate the virtual environment using the appropriate command for your operating system and environment:
```bash
# Mac / Linux
source .venv/bin/activate

# Windows CMD:
.venv\Scripts\activate.bat

# Windows PowerShell:
.venv\Scripts\Activate.ps1
```

Install ADK:
```bash
pip install google-adk
```

## Quick Start

### 1. Create Project
You will need to create a project structure in a parent folder:
```bash
mkdir -p weather_agent_project && cd weather_agent_project
python -m venv .venv
# Mac / Linux
source .venv/bin/activate
pip install google-adk
```

### 2. Create Your Agent

Run the `adk create` command to start a new agent project.
```bash
adk create weather_agent
```

The created agent project has the following structure, with the agent.py file containing the main control code for the agent.

```
weather_agent_project/
    weather_agent/
        agent.py      # main agent code
        .env          # API keys or project IDs
        __init__.py
```

```python
# weather_agent/agent.py
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

Run your agent using the `adk run` command-line tool and chat.
```bash
adk run weather_agent
# Opens http://localhost:8080 - select your agent and chat

# Or use replay file
adk run weather_agent --replay test.json
```

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
adk run weather_agent/ --replay test.json
```

### Model Selection

| Model | Use For |
|-------|---------|
| `gemini-3-flash-preview` | Most tasks (default) |
| `gemini-3-pro-preview` | Complex reasoning (requires `GOOGLE_CLOUD_LOCATION=global`) |
| `gemini-2.5-flash-lite` | High volume, cost-sensitive |

## Additional Resources

- **Official Documentation**: https://github.com/google/adk-docs
- **Repository**: https://github.com/google/adk-python
- **API Reference**: https://github.com/google/adk-docs/blob/main/docs/api-reference/python/