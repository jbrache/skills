# Configuration

Comprehensive guide to configuring Google ADK agents for different use cases and environments.

## Agent Configuration

### Basic Agent Parameters

```python
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="my_assistant",                 # Agent identifier (required)
    model="gemini-3-flash-preview",     # Model to use (required)
    instruction="You are a helpful...", # System-level guidance (required)
    description="Agent that...",        # Human-readable description (optional)
    tools=[],                           # List of available tools (optional)
    sub_agents=[],                      # Sub-agents for delegation (optional)
    response_model=None,                # Pydantic model for structured output (optional)
    callbacks=[]                        # Callback handlers (optional)
)
```

**Required parameters:**
- `name`: Unique identifier for the agent
- `model`: Gemini model to use (see Model Selection below)
- `instruction`: System-level guidance that shapes agent behavior

**Optional parameters:**
- `description`: Human-readable description for multi-agent coordination
- `tools`: List of tools the agent can use
- `sub_agents`: Specialized agents for delegation
- `response_model`: Pydantic schema for structured JSON output
- `callbacks`: List of callback handlers for observability

### Model Selection

Choose the right model based on your use case:

| Model | Best For | Speed | Cost | Max Tokens | Location |
|-------|----------|-------|------|------------|----------|
| **gemini-3-pro-preview** | Latest features, complex reasoning | Medium | Higher | 2M in / 8K out | `global` only |
| **gemini-3-flash-preview** | Fast preview features, high throughput | Fast | Medium | 1M in / 8K out | All regions |
| **gemini-2.5-flash-lite** | Ultra-fast, cost-effective tasks | Very Fast | Very Low | 1M in / 8K out | All regions |

```python
# Complex reasoning tasks
agent = LlmAgent(
    name="analyst",
    model="gemini-3-pro-preview",  # Requires location='global'
    instruction="Analyze complex data and provide insights."
)

# High-throughput applications
agent = LlmAgent(
    name="chatbot",
    model="gemini-3-flash-preview",  # Balanced speed and capability
    instruction="Respond to user questions quickly."
)

# Cost-sensitive tasks
agent = LlmAgent(
    name="classifier",
    model="gemini-2.5-flash-lite",  # Lowest cost
    instruction="Classify user input into categories."
)
```

**Important notes:**
- `gemini-3-pro-preview` requires `location='global'` in Vertex AI client initialization
- Choose based on: task complexity, latency requirements, cost constraints

### Instructions vs System Prompts

The `instruction` parameter shapes agent behavior system-wide:

```python
# Good: Clear, specific, actionable
agent = LlmAgent(
    name="customer_support",
    model="gemini-3-flash-preview",
    instruction=(
        "You are a helpful customer support agent. "
        "Always be polite and professional. "
        "If you cannot help, escalate to a human agent. "
        "Use the search_knowledge_base tool to find answers."
    )
)

# Avoid: Too vague
agent = LlmAgent(
    name="assistant",
    model="gemini-3-flash-preview",
    instruction="You are helpful."  # Too generic
)
```

**Best practices:**
- Be specific about agent role and behavior
- Include tool usage guidance if applicable
- Specify tone and communication style
- Set boundaries (what agent should/shouldn't do)

### Description for Multi-Agent Systems

The `description` parameter helps coordinator agents route requests:

```python
# Specialized agents with clear descriptions
researcher = LlmAgent(
    name="researcher",
    model="gemini-3-flash-preview",
    instruction="Research topics using web search.",
    description="Researches topics and gathers information from web sources"
)

analyst = LlmAgent(
    name="analyst",
    model="gemini-3-flash-preview",
    instruction="Analyze data and provide insights.",
    description="Analyzes data, creates visualizations, and provides insights"
)

# Coordinator uses descriptions to route
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-3-flash-preview",
    instruction="Route user requests to appropriate specialists.",
    description="Coordinates between research and analysis agents",
    sub_agents=[researcher, analyst]
)
```

## Sessions & Multi-Turn Conversations

Sessions track individual conversations and maintain context across multiple turns.

### Basic Multi-Turn Conversation

```python
from google.adk.agents import LlmAgent
from google.adk.sessions import Session

agent = LlmAgent(
    name="chat_assistant",
    model="gemini-3-flash-preview",
    instruction="You are a helpful assistant. Remember conversation context."
)

# Create session for this conversation
session = Session(session_id="user-123")

# First turn
response1 = agent.run(input="My name is Alice", session=session)
# Output: "Nice to meet you, Alice!"

# Second turn - agent remembers previous context
response2 = agent.run(input="What's my name?", session=session)
# Output: "Your name is Alice."
```

**Key points:**
- Create a `Session` with a unique `session_id` (e.g., user ID, conversation ID)
- Pass the same session to all turns in the conversation
- ADK automatically maintains conversation history

### Multiple Conversations

```python
# Different users use different sessions
alice_session = Session(session_id="alice")
bob_session = Session(session_id="bob")

agent.run(input="I like pizza", session=alice_session)
agent.run(input="I like sushi", session=bob_session)

# Each session maintains its own history
agent.run(input="What do I like?", session=alice_session)
# Output: "You mentioned you like pizza."
```

---

## Callbacks

Callbacks enable observability, customization, and control over agent behavior.

```python
from google.adk.callbacks import CallbackHandler

class CustomCallback(CallbackHandler):
    def on_agent_start(self, agent_name: str, input_data: dict):
        print(f"Agent {agent_name} started")

    def on_agent_end(self, agent_name: str, output: dict):
        print(f"Agent {agent_name} completed")

    def on_tool_start(self, tool_name: str, inputs: dict):
        print(f"Tool {tool_name} called")

    def on_tool_end(self, tool_name: str, output: str):
        print(f"Tool {tool_name} returned")

    def on_error(self, error: Exception):
        print(f"Error: {error}")

agent = LlmAgent(
    name="monitored_assistant",
    model="gemini-3-flash-preview",
    callbacks=[CustomCallback()]
)
```

### Observability Integration

```python
# Arize AX integration
from google.adk.observability import ArizeCallback

agent = LlmAgent(
    callbacks=[ArizeCallback(api_key="...", space_id="...")]
)

# Phoenix integration
from google.adk.observability import PhoenixCallback

agent = LlmAgent(
    callbacks=[PhoenixCallback(endpoint="http://localhost:6006")]
)
```

---

## Runtime Configuration

### RunConfig Parameters

Control agent behavior at runtime:

```python
from google.adk.runtime import RunConfig

config = RunConfig(
    max_turns=10,                  # Maximum conversation turns
    timeout=30,                    # Timeout per turn (seconds)
    streaming=False,               # Enable streaming responses
    enable_observability=True,     # Enable built-in observability
    retry_config={...},           # Retry configuration (see below)
    safety_settings={...},        # Safety settings (see below)
    generation_config={...}       # Generation parameters (see below)
)

# Apply to agent run
response = agent.run(input="Query", config=config)
```

### Maximum Turns

Prevent infinite loops in multi-turn conversations:

```python
# Default: No limit (can loop indefinitely)
config = RunConfig(max_turns=10)

# Agent stops after 10 turns, even if not resolved
response = agent.run(input="Complex task", config=config)
```

**When to use:**
- Multi-turn conversations with tools
- Prevent runaway agent loops
- Control costs in production

### Timeout Configuration

Set per-turn timeout limits:

```python
# 30-second timeout per turn
config = RunConfig(timeout=30)

try:
    response = agent.run(input="Long-running task", config=config)
except TimeoutError:
    print("Agent timed out")
```

**Recommendations:**
- Simple tasks: 10-15 seconds
- Tool-heavy tasks: 30-60 seconds
- Complex reasoning: 60-120 seconds

### Streaming

Enable real-time response streaming:

```python
# Enable streaming
config = RunConfig(streaming=True)

# Stream responses
for chunk in agent.stream(input="Tell me a story", config=config):
    print(chunk, end="", flush=True)
```

**Important limitations:**
- **NOT** compatible with function calling (tools)
- **NOT** compatible with structured outputs (response_model)
- Use async streaming for better performance: `agent.astream()`

See [Streaming](../SKILL.md#streaming-responses) for detailed examples.

## Generation Parameters

### Temperature

Controls randomness in responses:

```python
config = RunConfig(
    generation_config={
        "temperature": 0.7  # Range: 0.0 to 2.0
    }
)
```

**Temperature values:**
- `0.0`: Deterministic, focused (best for factual tasks)
- `0.7`: Balanced creativity and consistency (default, general use)
- `1.0`: More creative and varied
- `1.5-2.0`: Highly creative, less predictable

**Use cases:**
- `0.0-0.3`: Code generation, data extraction, classification
- `0.5-0.9`: Conversations, Q&A, general assistance
- `1.0-2.0`: Creative writing, brainstorming, ideation

### Top-P (Nucleus Sampling)

Controls diversity by sampling from top probability mass:

```python
config = RunConfig(
    generation_config={
        "top_p": 0.9  # Range: 0.0 to 1.0
    }
)
```

**Top-P values:**
- `0.1-0.5`: Focused, conservative outputs
- `0.9`: Balanced (default)
- `0.95-1.0`: More diverse outputs

**Recommendation:** Use either temperature OR top_p, not both.

### Top-K

Limits sampling to top K most likely tokens:

```python
config = RunConfig(
    generation_config={
        "top_k": 40  # Typical range: 1 to 100
    }
)
```

**Top-K values:**
- `1`: Greedy decoding (always picks most likely token)
- `40`: Balanced (default)
- `100+`: More diversity

### Max Output Tokens

Limit response length:

```python
config = RunConfig(
    generation_config={
        "max_output_tokens": 2048  # Maximum tokens in response
    }
)
```

**Model limits:**
- `gemini-3-pro-preview`: Up to 8,192 tokens
- `gemini-3-flash-preview`: Up to 8,192 tokens
- `gemini-2.5-flash-lite`: Up to 8,192 tokens

**Use cases:**
- Short responses: 256-512 tokens
- Medium responses: 512-1024 tokens
- Long-form content: 2048-4096 tokens

### Complete Generation Config

```python
config = RunConfig(
    generation_config={
        "temperature": 0.7,        # Creativity level
        "top_p": 0.9,             # Nucleus sampling
        "top_k": 40,              # Token sampling limit
        "max_output_tokens": 2048 # Response length limit
    }
)

response = agent.run(input="Generate content", config=config)
```

## Safety Settings

### Safety Thresholds

Configure content filtering:

```python
config = RunConfig(
    safety_settings={
        "harassment": "BLOCK_MEDIUM_AND_ABOVE",
        "hate_speech": "BLOCK_MEDIUM_AND_ABOVE",
        "sexually_explicit": "BLOCK_MEDIUM_AND_ABOVE",
        "dangerous_content": "BLOCK_MEDIUM_AND_ABOVE"
    }
)
```

**Threshold levels:**
- `BLOCK_NONE`: No blocking
- `BLOCK_ONLY_HIGH`: Block only high-severity content
- `BLOCK_MEDIUM_AND_ABOVE`: Block medium and high severity (recommended)
- `BLOCK_LOW_AND_ABOVE`: Strictest filtering

**Categories:**
- `harassment`: Harassment and bullying
- `hate_speech`: Hateful or discriminatory content
- `sexually_explicit`: Sexual content
- `dangerous_content`: Dangerous or harmful activities

### Production Safety Configuration

```python
# Strict safety for production
config = RunConfig(
    safety_settings={
        "harassment": "BLOCK_MEDIUM_AND_ABOVE",
        "hate_speech": "BLOCK_MEDIUM_AND_ABOVE",
        "sexually_explicit": "BLOCK_MEDIUM_AND_ABOVE",
        "dangerous_content": "BLOCK_MEDIUM_AND_ABOVE"
    }
)

# Development/testing (more permissive)
dev_config = RunConfig(
    safety_settings={
        "harassment": "BLOCK_ONLY_HIGH",
        "hate_speech": "BLOCK_ONLY_HIGH",
        "sexually_explicit": "BLOCK_ONLY_HIGH",
        "dangerous_content": "BLOCK_ONLY_HIGH"
    }
)
```

**Custom content filters:**
```python
from google.adk.safety import SafetySettings

safety = SafetySettings(
    custom_filters=[
        {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "block": True}  # SSN pattern
    ]
)
```

## Retry Configuration

### Automatic Retry Settings

Configure retry behavior for transient errors:

```python
config = RunConfig(
    retry_config={
        "max_retries": 3,           # Maximum retry attempts
        "backoff_factor": 2,        # Exponential backoff multiplier
        "initial_wait": 1.0,        # Initial wait time (seconds)
        "max_wait": 60.0,          # Maximum wait time (seconds)
        "retry_on": [               # Errors to retry
            "ResourceExhausted",
            "ServiceUnavailable",
            "DeadlineExceeded"
        ]
    }
)
```

**Parameters:**
- `max_retries`: Number of retry attempts (recommended: 3-5)
- `backoff_factor`: Multiplier for exponential backoff (recommended: 2)
- `initial_wait`: First retry delay in seconds (recommended: 1.0)
- `max_wait`: Cap on retry delay (recommended: 60.0)
- `retry_on`: List of error types to retry

**Retry timing:**
- Attempt 1: immediate
- Attempt 2: wait 1s (initial_wait × backoff_factor^0)
- Attempt 3: wait 2s (initial_wait × backoff_factor^1)
- Attempt 4: wait 4s (initial_wait × backoff_factor^2)

See [Error Handling](error-handling.md#retry-configuration) for manual retry patterns.

## Complete Configuration Example

### Production-Ready Configuration

```python
from google.adk.agents import LlmAgent
from google.adk.runtime import RunConfig
from google.adk.tools import google_search

# Configure agent
agent = LlmAgent(
    name="production_assistant",
    model="gemini-3-flash-preview",
    instruction=(
        "You are a professional customer support agent. "
        "Be helpful, concise, and accurate. "
        "Use tools to find information when needed."
    ),
    description="Customer support agent with search capabilities",
    tools=[google_search]
)

# Production runtime configuration
config = RunConfig(
    # Conversation limits
    max_turns=10,
    timeout=30,

    # Performance
    streaming=False,  # Disabled (using tools)
    enable_observability=True,

    # Reliability
    retry_config={
        "max_retries": 3,
        "backoff_factor": 2,
        "initial_wait": 1.0,
        "max_wait": 60.0
    },

    # Safety
    safety_settings={
        "harassment": "BLOCK_MEDIUM_AND_ABOVE",
        "hate_speech": "BLOCK_MEDIUM_AND_ABOVE",
        "sexually_explicit": "BLOCK_MEDIUM_AND_ABOVE",
        "dangerous_content": "BLOCK_MEDIUM_AND_ABOVE"
    },

    # Generation quality
    generation_config={
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 1024
    }
)

# Run with configuration
response = agent.run(
    input="How do I reset my password?",
    config=config
)
```

## Environment Variables

### GCP Configuration

```bash
# Required
export GOOGLE_CLOUD_PROJECT=my-project-id
export GOOGLE_CLOUD_LOCATION=us-central1

# Optional (if not using gcloud auth)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# For gemini-3-pro-preview
export GOOGLE_CLOUD_LOCATION=global
```

### Application Configuration

```bash
# Agent settings
export AGENT_MODEL=gemini-3-flash-preview
export AGENT_MAX_TURNS=10
export AGENT_TIMEOUT=30

# Feature flags
export ENABLE_STREAMING=false
export ENABLE_OBSERVABILITY=true

# Safety
export SAFETY_LEVEL=BLOCK_MEDIUM_AND_ABOVE
```

### Loading Environment Variables

```python
import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()

# Use in configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = os.environ.get("AGENT_MODEL", "gemini-3-flash-preview")

agent = LlmAgent(
    name="env_configured_assistant",
    model=MODEL,
    instruction="Environment-configured agent"
)
```

## Vertex AI Authentication

### Using Vertex AI with Application Default Credentials

When using Vertex AI backend instead of the Google AI API, configure your `.env` file:

```bash
# .env file for Vertex AI
# CRITICAL: These exact environment variables are required

# Use Vertex AI backend (not Google AI API)
GOOGLE_GENAI_USE_VERTEXAI=true

# Location (use 'global' for preview models)
GOOGLE_CLOUD_LOCATION=global

# Optional: Specify project (uses default if not set)
# GOOGLE_CLOUD_PROJECT=your-project-id
```

**Important notes:**
- `GOOGLE_GENAI_USE_VERTEXAI=true` tells the SDK to use Vertex AI instead of Google AI API
- `GOOGLE_CLOUD_LOCATION=global` is required for preview models like `gemini-3-pro-preview`
- Project ID is automatically detected from application default credentials if not specified

### Common Authentication Patterns

**Pattern 1: Application Default Credentials (Recommended)**

```bash
# Authenticate once
gcloud auth application-default login

# Set default project (optional, but recommended)
gcloud config set project your-project-id
```

Then in your `.env`:
```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_LOCATION=global
# No need for GOOGLE_CLOUD_PROJECT if default is set
```

**Pattern 2: Service Account**

```bash
# .env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_LOCATION=global
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**Pattern 3: Explicit Project**

```bash
# .env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_LOCATION=global
GOOGLE_CLOUD_PROJECT=specific-project-id
```

### Authentication Errors and Solutions

**Error: Missing authentication**
```
ValueError: Missing key inputs argument! To use the Google AI API, provide (`api_key`)
arguments. To use the Google Cloud API, provide (`vertexai`, `project` & `location`) arguments.
```

**Solution:** Add to `.env`:
```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_LOCATION=global
```

**Error: Model not available in region**
```
Error: Model gemini-3-pro-preview not available in us-central1
```

**Solution:** Use `global` location for preview models:
```bash
GOOGLE_CLOUD_LOCATION=global
```

### Model-Specific Location Requirements

Different models have different regional availability:

| Model | Required Location | Notes |
|-------|-------------------|-------|
| `gemini-3-pro-preview` | `global` | Only available in global location |
| `gemini-3-flash-preview` | Any region or `global` | Available in all regions |
| `gemini-2.5-flash-lite` | Any region | Available in all regions |

Example configuration by model:

```python
# For gemini-3-pro-preview
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
agent = LlmAgent(model="gemini-3-pro-preview", ...)

# For gemini-3-flash-preview (regional)
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
agent = LlmAgent(model="gemini-3-flash-preview", ...)

# For gemini-3-flash-preview (global)
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
agent = LlmAgent(model="gemini-3-flash-preview", ...)
```

### Complete Vertex AI Setup Example

```python
# setup.py
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent

# Load environment variables
load_dotenv()

# Verify Vertex AI configuration
assert os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "true", \
    "Must set GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI"

location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
print(f"Using Vertex AI in location: {location}")

# Create agent
agent = LlmAgent(
    name="vertex_assistant",
    model="gemini-3-flash-preview",
    instruction="Agent using Vertex AI backend"
)

# Test authentication
try:
    response = agent.run(input="Hello")
    print("✓ Authentication successful")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
    print("Run: gcloud auth application-default login")
```

## Configuration Presets

### Quick Configuration Presets

```python
# Fast responses (low latency)
FAST_CONFIG = RunConfig(
    timeout=10,
    generation_config={"temperature": 0.5, "max_output_tokens": 512}
)

# High quality (best responses)
QUALITY_CONFIG = RunConfig(
    timeout=60,
    generation_config={"temperature": 0.7, "max_output_tokens": 2048}
)

# Cost-effective (minimize costs)
ECONOMICAL_CONFIG = RunConfig(
    timeout=15,
    generation_config={"temperature": 0.5, "max_output_tokens": 512}
)

# Creative (brainstorming)
CREATIVE_CONFIG = RunConfig(
    timeout=30,
    generation_config={"temperature": 1.2, "max_output_tokens": 2048}
)

# Use preset
response = agent.run(input="Query", config=FAST_CONFIG)
```

## Configuration Best Practices

### Development vs Production

```python
import os

# Environment-specific configuration
if os.environ.get("ENV") == "production":
    config = RunConfig(
        max_turns=10,
        timeout=30,
        enable_observability=True,
        safety_settings={
            "harassment": "BLOCK_MEDIUM_AND_ABOVE",
            "hate_speech": "BLOCK_MEDIUM_AND_ABOVE",
            "sexually_explicit": "BLOCK_MEDIUM_AND_ABOVE",
            "dangerous_content": "BLOCK_MEDIUM_AND_ABOVE"
        },
        retry_config={"max_retries": 3, "backoff_factor": 2}
    )
else:
    # More permissive for development
    config = RunConfig(
        max_turns=5,
        timeout=15,
        enable_observability=False,
        safety_settings={
            "harassment": "BLOCK_ONLY_HIGH",
            "hate_speech": "BLOCK_ONLY_HIGH",
            "sexually_explicit": "BLOCK_ONLY_HIGH",
            "dangerous_content": "BLOCK_ONLY_HIGH"
        }
    )
```

### Configuration Validation

```python
def validate_config(config: RunConfig):
    """Validate runtime configuration."""
    # Check timeout
    if config.timeout and config.timeout < 5:
        raise ValueError("Timeout too low (minimum 5 seconds)")

    # Check max_turns
    if config.max_turns and config.max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    # Validate generation config
    if config.generation_config:
        temp = config.generation_config.get("temperature", 0.7)
        if not 0.0 <= temp <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")

    return True

# Use validation
validate_config(config)
response = agent.run(input="Query", config=config)
```

## Additional Resources

- **Runtime Configuration**: https://github.com/google/adk-docs/blob/main/docs/runtime/index.md
- **Safety Settings**: https://github.com/google/adk-docs/blob/main/docs/safety/index.md
- **Generation Parameters**: https://ai.google.dev/gemini-api/docs/models/generative-models#model-parameters
- **Model Selection Guide**: ../SKILL.md#model-selection
