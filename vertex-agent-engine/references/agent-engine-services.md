# Agent Engine Services

Vertex AI Agent Engine provides managed services that enhance agent capabilities in production environments. This reference covers all key services.

## Table of Contents

- [Sessions](#sessions)
- [Memory Bank](#memory-bank)
- [Code Execution](#code-execution)
- [Example Store](#example-store)
- [Observability](#observability)

## Sessions

Sessions maintain the history of interactions between users and agents, providing conversation context and state management.

### Overview

A session represents the chronological sequence of messages and actions (events) for a single, ongoing interaction between a user and your agent system. Sessions provide:

- **Conversation history** - Full record of user queries and agent responses
- **State management** - Temporary data relevant only during current conversation
- **Context continuity** - Maintain context across multiple turns
- **Rewind capability** - Roll back to any point in conversation (GA Dec 2024)

**Status**: Generally Available (GA) as of December 2024

### Creating and Managing Sessions

```python
from google.cloud import aiplatform
from google import genai

# Initialize client
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Create a session for multi-turn conversation
session = client.sessions.create(
    agent_engine_name="projects/PROJECT/locations/LOCATION/reasoning_engines/AGENT_ID",
    display_name="Customer support chat - User 12345"
)

print(f"Session created: {session.name}")

# Send first message
response1 = session.query("What's the weather in San Francisco?")
print(response1.text)

# Continue conversation with context
response2 = session.query("What about tomorrow?")  # Knows we're talking about SF
print(response2.text)

# Get session history
history = session.get_history()
for event in history.events:
    print(f"{event.role}: {event.content}")
```

### Session Operations

**List all sessions:**

```python
# List sessions for an agent
sessions = client.sessions.list(
    parent="projects/PROJECT/locations/LOCATION/reasoning_engines/AGENT_ID"
)

for session in sessions:
    print(f"Session: {session.display_name}, Created: {session.create_time}")
```

**Resume an existing session:**

```python
# Get existing session by name
session = client.sessions.get(
    name="projects/PROJECT/locations/LOCATION/reasoning_engines/AGENT_ID/sessions/SESSION_ID"
)

# Continue conversation
response = session.query("Can you summarize our conversation so far?")
```

**Rewind a session (remove recent interactions):**

```python
# Rewind to a specific event
session.rewind(event_id="event_xyz")

# This invalidates all interactions after that point
# Useful for removing incorrect or unwanted context
```

**Delete a session:**

```python
# Delete when conversation is complete
client.sessions.delete(session.name)
```

### Session State Management

Sessions can store temporary state data:

```python
# Update session state
session.update_state({
    "user_preference": "metric units",
    "language": "English",
    "context": "discussing weather patterns"
})

# Access state in agent tools
def get_weather(location: str, session_state: dict) -> str:
    """Get weather using user's preferred units."""
    units = session_state.get("user_preference", "imperial")
    # Fetch weather data...
    return f"Weather in {location}: 72°F (metric: 22°C)" if units == "metric" else "Weather in {location}: 72°F"
```

### Automatic Session Management with ADK

ADK automatically manages sessions when using the agent query interface:

```python
from google import genai

agent = genai.Agent(
    model="gemini-2.5-flash",
    tools=[...],
    system_instruction="You are a helpful assistant."
)

# Deploy to Agent Engine
remote_agent = client.agent_engines.create(agent, config={...})

# ADK automatically creates and manages sessions
# Each query automatically maintains context
response1 = remote_agent.query("My name is Alice")
response2 = remote_agent.query("What's my name?")  # Agent remembers "Alice"
```

## Memory Bank

Memory Bank provides persistent, personalized memory across sessions, enabling agents to remember user preferences, facts, and context over time.

### Overview

Memory Bank transforms stateless agent interactions into stateful, contextual experiences where the agent remembers, learns, and adapts. Key features:

- **Cross-session persistence** - Memories persist across multiple conversations
- **Automatic extraction** - Gemini models analyze conversations to extract memories
- **Personalization** - Each user has their own memory bank
- **Asynchronous updates** - Memory extraction happens in the background

**Status**: Generally Available (GA) as of December 2024

### Use Cases

- **Personalized recommendations** - Remember user preferences and history
- **Continuous learning** - Build knowledge about users over time
- **Context awareness** - Recall previous conversations and decisions
- **Improved UX** - Users don't need to repeat information

### Setting Up Memory Bank

```python
from google.cloud import aiplatform
from vertexai.preview import rag

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Create a Memory Bank for a user
memory_bank = rag.create_corpus(
    display_name=f"memory_bank_user_{user_id}",
    description="Personal memory bank for user preferences and context"
)

print(f"Memory Bank created: {memory_bank.name}")
```

### Storing Memories

Memories can be added manually or extracted automatically from conversations:

**Manual memory storage:**

```python
from vertexai.preview import rag

# Add specific memories
memories = [
    "User prefers metric units for temperature",
    "User is interested in machine learning and AI",
    "User's timezone is PST (UTC-8)",
    "User has two dogs named Max and Bella"
]

# Store memories in the corpus
for memory in memories:
    rag.import_files(
        corpus_name=memory_bank.name,
        paths=[{"inline_data": memory}],
        chunk_size=512
    )
```

**Automatic extraction from conversations:**

```python
# Configure agent to automatically extract memories
agent = genai.Agent(
    model="gemini-2.5-flash",
    tools=[...],
    config={
        "memory_bank": {
            "corpus_name": memory_bank.name,
            "auto_extract": True,  # Automatically extract memories
            "extraction_model": "gemini-2.5-flash"
        }
    }
)

# As users interact, memories are automatically extracted
# Example conversation:
# User: "I prefer dark mode and I'm a vegetarian"
# Agent response generated...
# Background: System extracts memories:
# - "User prefers dark mode"
# - "User is vegetarian"
```

### Retrieving Memories

Memories are automatically retrieved and injected into agent context:

```python
def create_agent_with_memory(user_id: str):
    """Create agent with user's memory bank."""

    # Retrieval function for memories
    def retrieve_user_context(query: str) -> str:
        """Retrieve relevant memories for the query."""
        results = rag.retrieval_query(
            corpus_name=f"memory_bank_user_{user_id}",
            text=query,
            similarity_top_k=5
        )
        memories = "\n".join([r.text for r in results.contexts])
        return memories

    # Agent uses memories to personalize responses
    agent = genai.Agent(
        model="gemini-2.5-flash",
        tools=[retrieve_user_context, ...],
        system_instruction="""You are a personalized assistant.
        Before responding, retrieve relevant user context and memories.
        Use this information to personalize your responses."""
    )

    return agent
```

### Managing Memories

**List memories:**

```python
# Query all memories for a user
results = rag.retrieval_query(
    corpus_name=memory_bank.name,
    text="",  # Empty query returns all
    similarity_top_k=100
)

for memory in results.contexts:
    print(f"Memory: {memory.text}")
```

**Update memories:**

```python
# Memories are immutable - add new version
new_memory = "User now prefers imperial units (changed from metric)"
rag.import_files(
    corpus_name=memory_bank.name,
    paths=[{"inline_data": new_memory}]
)
```

**Delete memories:**

```python
# Delete specific file/memory
rag.delete_file(file_name="FILE_ID")

# Or delete entire memory bank
rag.delete_corpus(name=memory_bank.name)
```

### Memory Bank Best Practices

1. **Namespace by user** - Create separate memory banks per user
2. **Privacy controls** - Implement access controls and encryption
3. **Memory pruning** - Periodically review and remove outdated memories
4. **Explicit consent** - Get user permission before storing personal information
5. **Transparency** - Let users view and manage their memories

## Code Execution

Code Execution enables agents to run Python or JavaScript code in a secure, isolated sandbox environment.

### Overview

Agent Engine Code Execution provides:

- **Secure sandbox** - Isolated environment with no network access
- **Language support** - Python 3.11+ and JavaScript (Node.js)
- **File system** - Limited file system (100MB max)
- **Compute resources** - 2 vCPU / 1.5GB RAM (standard) or 4 vCPU / 4GB RAM
- **Framework agnostic** - Works with any agent framework

**Status**: Preview

### Use Cases

- **Data analysis** - Run pandas, numpy for data processing
- **Mathematical computation** - Complex calculations and simulations
- **Code generation and testing** - Generate and execute code snippets
- **Dynamic tool creation** - Create tools on-the-fly based on user needs

### Setting Up Code Execution

```python
from google.cloud import aiplatform
from vertexai import Client
from vertexai.preview import code_execution

# Initialize
client = Client(project=PROJECT_ID, location=LOCATION)

# Create a code execution sandbox
sandbox = client.code_execution.create_sandbox(
    display_name="My Agent Sandbox",
    config={
        "language": "LANGUAGE_PYTHON",  # or LANGUAGE_JAVASCRIPT
        "machine_type": "MACHINE_CONFIG_VCPU2_RAM1_5GIB"  # or MACHINE_CONFIG_VCPU4_RAM4GIB
    }
)

print(f"Sandbox created: {sandbox.name}")
```

### Executing Code

**Basic code execution:**

```python
# Execute Python code
result = sandbox.execute(
    code="""
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'x': np.arange(10),
    'y': np.random.randn(10)
})

# Calculate statistics
mean_y = data['y'].mean()
std_y = data['y'].std()

print(f"Mean: {mean_y:.2f}, Std: {std_y:.2f}")
"""
)

print(f"Output: {result.output}")
print(f"Errors: {result.errors}")
```

**With file inputs:**

```python
# Execute code with input files
result = sandbox.execute(
    code="""
import pandas as pd

# Read CSV file
df = pd.read_csv('data.csv')
print(df.describe())
""",
    files={
        "data.csv": "name,age,score\nAlice,25,90\nBob,30,85\nCarol,28,95"
    }
)

print(result.output)
```

**Retrieve output files:**

```python
# Execute code that generates files
result = sandbox.execute(
    code="""
import json

data = {'result': 42, 'status': 'success'}

with open('output.json', 'w') as f:
    json.dump(data, f)

print('File created')
"""
)

# Access generated files
if result.output_files:
    output_json = result.output_files.get('output.json')
    print(f"Output file content: {output_json}")
```

### Integrating with Agents

**Create an agent with code execution capability:**

```python
from google import genai

def execute_python(code: str) -> str:
    """Execute Python code in a secure sandbox.

    Args:
        code: Python code to execute

    Returns:
        Execution output or error message
    """
    try:
        result = sandbox.execute(code=code)
        if result.errors:
            return f"Error: {result.errors}"
        return result.output
    except Exception as e:
        return f"Execution failed: {str(e)}"


agent = genai.Agent(
    model="gemini-2.5-pro",
    tools=[execute_python],
    system_instruction="""You are a data analysis assistant with code execution capabilities.

    When users ask for data analysis or calculations:
    1. Write clear, efficient Python code
    2. Use pandas, numpy, matplotlib as needed
    3. Execute the code using execute_python tool
    4. Explain the results to the user

    Always validate inputs and handle errors gracefully."""
)
```

**Example agent interaction:**

```python
# Deploy agent
remote_agent = client.agent_engines.create(agent, config={...})

# User query triggers code execution
response = remote_agent.query("""
I have sales data: [100, 150, 200, 180, 220, 250].
Calculate the mean, median, and growth rate.
""")

# Agent generates and executes:
# import numpy as np
# sales = [100, 150, 200, 180, 220, 250]
# mean = np.mean(sales)
# median = np.median(sales)
# growth = (sales[-1] - sales[0]) / sales[0] * 100
# print(f"Mean: {mean}, Median: {median}, Growth: {growth}%")

print(response.text)  # Agent explains the results
```

### Code Execution Limitations

- **No network access** - Cannot make HTTP requests or access external APIs
- **File size limits** - Max 100MB per request/response
- **Time limits** - Execution timeout (typically 60 seconds)
- **Package restrictions** - Only pre-installed packages available
- **Stateless** - Sandbox state resets between executions (unless explicitly preserved)

### Pre-installed Packages

**Python packages include:**
- pandas, numpy, scipy
- matplotlib, seaborn (plotting)
- scikit-learn (machine learning)
- requests (limited use)

**For custom packages**, contact Google Cloud support to request additions.

## Example Store

Example Store enables few-shot learning by storing and dynamically retrieving input-output examples to improve agent performance.

### Overview

Example Store helps agents learn from examples by:

- **Demonstrating patterns** - Show expected behavior through examples
- **Improving accuracy** - Few-shot learning improves response quality
- **Function calling guidance** - Demonstrate correct tool usage
- **Consistency** - Ensure consistent response formatting

**Status**: Preview

### Use Cases

- **Function calling** - Show which functions to call for specific queries
- **Output formatting** - Demonstrate expected response structure
- **Domain adaptation** - Provide domain-specific examples
- **Edge case handling** - Show how to handle unusual inputs

### Creating an Example Store

```python
from google.cloud import aiplatform
from vertexai.preview import examples

# Create example store
example_store = examples.create_store(
    display_name="Customer Support Examples",
    description="Few-shot examples for customer support agent"
)

print(f"Example Store created: {example_store.name}")
```

### Adding Examples

Examples consist of input-output pairs:

```python
# Define few-shot examples
training_examples = [
    {
        "input": "I want to return my order",
        "output": {
            "function_call": "create_return_request",
            "explanation": "User wants to initiate a return",
            "response": "I can help you return your order. Let me create a return request for you."
        }
    },
    {
        "input": "Where is my package?",
        "output": {
            "function_call": "check_order_status",
            "explanation": "User asking about delivery status",
            "response": "Let me check the status of your order for you."
        }
    },
    {
        "input": "Do you ship to Canada?",
        "output": {
            "function_call": "search_knowledge_base",
            "parameters": {"category": "shipping", "query": "Canada"},
            "response": "Let me check our shipping policies for Canada."
        }
    }
]

# Add examples to store
for example in training_examples:
    examples.add_example(
        store_name=example_store.name,
        input_text=example["input"],
        output_data=example["output"]
    )
```

### Retrieving Examples for Agent Context

```python
def retrieve_relevant_examples(query: str, top_k: int = 3) -> list:
    """Retrieve most relevant examples for a query.

    Args:
        query: User query
        top_k: Number of examples to retrieve

    Returns:
        List of relevant examples
    """
    results = examples.search(
        store_name=example_store.name,
        query=query,
        top_k=top_k
    )

    return [
        {
            "input": r.input_text,
            "output": r.output_data
        }
        for r in results
    ]
```

### Integrating with Agents

```python
from google import genai

def create_agent_with_examples():
    """Create agent that uses Example Store for few-shot learning."""

    def get_examples_for_query(query: str) -> str:
        """Retrieve and format relevant examples."""
        examples_list = retrieve_relevant_examples(query, top_k=3)

        formatted = "Here are similar examples:\n\n"
        for i, ex in enumerate(examples_list, 1):
            formatted += f"Example {i}:\n"
            formatted += f"Input: {ex['input']}\n"
            formatted += f"Output: {ex['output']}\n\n"

        return formatted

    agent = genai.Agent(
        model="gemini-2.5-flash",
        tools=[check_order_status, create_return_request, get_examples_for_query],
        system_instruction="""You are a customer support agent.

        For each user query:
        1. Retrieve relevant examples using get_examples_for_query
        2. Follow the pattern demonstrated in the examples
        3. Call the appropriate function based on the examples
        4. Provide helpful responses in a similar style

        Use examples to ensure consistency and accuracy."""
    )

    return agent
```

### Automatic Example Retrieval

Configure agents to automatically retrieve examples:

```python
agent = genai.Agent(
    model="gemini-2.5-flash",
    tools=[...],
    config={
        "example_store": {
            "store_name": example_store.name,
            "top_k": 3,
            "auto_retrieve": True  # Automatically inject examples
        }
    },
    system_instruction="Follow the patterns shown in the provided examples."
)
```

## Observability

Agent Engine provides built-in observability through Google Cloud's monitoring, logging, and tracing services.

### Overview

Observability features include:

- **Cloud Logging** - All queries, responses, and tool calls logged
- **Cloud Monitoring** - Metrics for latency, errors, token usage
- **Cloud Trace** - Distributed tracing for debugging
- **Cost tracking** - Token usage and billing metrics

### Cloud Logging

All agent interactions are automatically logged:

```python
# View logs in Cloud Console or via gcloud
# Navigate to: Logging > Logs Explorer

# Query logs programmatically
from google.cloud import logging

client = logging.Client(project=PROJECT_ID)
logger = client.logger("agent-engine")

# Fetch recent logs
for entry in logger.list_entries(page_size=10):
    print(f"[{entry.timestamp}] {entry.payload}")
```

**Log structure:**

```json
{
  "timestamp": "2025-12-28T10:30:00Z",
  "resource": {
    "type": "agent_engine",
    "labels": {
      "agent_id": "AGENT_ID",
      "session_id": "SESSION_ID"
    }
  },
  "jsonPayload": {
    "query": "What's the weather?",
    "response": "The weather in...",
    "tool_calls": ["get_weather"],
    "latency_ms": 1234,
    "token_count": {
      "input": 50,
      "output": 100
    }
  }
}
```

### Cloud Monitoring

Monitor agent performance with custom dashboards:

```python
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{PROJECT_ID}"

# Query agent latency metrics
query = f"""
fetch agent_engine::latency
| filter resource.agent_id == "AGENT_ID"
| group_by 5m, [value_latency_mean: mean(value.latency)]
"""

# Create custom metrics
series = monitoring_v3.TimeSeries()
series.metric.type = "custom.googleapis.com/agent/success_rate"
# ... configure and send
```

**Key metrics to monitor:**

- `agent_engine/query_count` - Number of queries
- `agent_engine/latency` - Response time
- `agent_engine/error_rate` - Error percentage
- `agent_engine/token_usage` - Token consumption
- `agent_engine/tool_call_count` - Tool invocations

### Cloud Trace

Distributed tracing for debugging agent workflows:

```python
from google.cloud import trace_v2

# Traces are automatically created for each query
# View in Cloud Console: Trace > Trace Explorer

# Access trace data programmatically
client = trace_v2.TraceServiceClient()
project_id = f"projects/{PROJECT_ID}"

traces = client.list_traces(name=project_id)
for trace in traces:
    print(f"Trace: {trace.trace_id}")
    for span in trace.spans:
        print(f"  Span: {span.name}, Duration: {span.end_time - span.start_time}")
```

### Setting Up Alerts

Configure alerts for production agents:

```python
from google.cloud import monitoring_v3

# Create alert policy for high error rate
alert_client = monitoring_v3.AlertPolicyServiceClient()

policy = monitoring_v3.AlertPolicy(
    display_name="Agent High Error Rate",
    conditions=[{
        "display_name": "Error rate > 5%",
        "condition_threshold": {
            "filter": 'resource.type="agent_engine" AND metric.type="agent_engine/error_rate"',
            "comparison": "COMPARISON_GT",
            "threshold_value": 0.05,
            "duration": {"seconds": 300}
        }
    }],
    notification_channels=[NOTIFICATION_CHANNEL_ID]
)

created_policy = alert_client.create_alert_policy(
    name=f"projects/{PROJECT_ID}",
    alert_policy=policy
)
```

### Cost Monitoring

Track token usage and costs:

```python
# Query cost metrics
from google.cloud import billing_v1

client = billing_v1.CloudBillingClient()

# Get billing account
billing_account = f"billingAccounts/{BILLING_ACCOUNT_ID}"

# Query costs by agent
# Use BigQuery export for detailed cost analysis
```

### Best Practices

1. **Enable all logging** - Don't disable logs to save costs; they're invaluable for debugging
2. **Set up alerts** - Alert on error rates, latency spikes, and quota limits
3. **Create dashboards** - Build custom dashboards for your team
4. **Monitor token usage** - Track costs and optimize prompts
5. **Use trace for debugging** - Leverage distributed tracing to understand complex agent workflows
6. **Retention policies** - Configure appropriate log retention (30-90 days typical)
