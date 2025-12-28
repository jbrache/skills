# Agent Starter Pack

The Agent Starter Pack is a production-ready framework for building and deploying GenAI agents to Google Cloud with built-in CI/CD, evaluation, and observability.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Creating an Agent](#creating-an-agent)
- [Agent Templates](#agent-templates)
- [Deployment](#deployment)
- [CI/CD Integration](#cicd-integration)
- [Evaluation](#evaluation)
- [Monitoring and Observability](#monitoring-and-observability)

## Overview

Agent Starter Pack provides:

- **Production-ready templates** - Pre-built agents (ReAct, RAG, multi-agent, Live API)
- **Built-in CI/CD** - Automated testing, deployment pipelines
- **Evaluation framework** - Test your agents before deployment
- **Observability** - Monitoring, logging, tracing out of the box
- **Deployment flexibility** - Deploy to Agent Engine or Cloud Run

Repository: https://github.com/GoogleCloudPlatform/agent-starter-pack

## Installation

```bash
# Install the CLI
pip install agent-starter-pack

# Or clone the repository
git clone https://github.com/GoogleCloudPlatform/agent-starter-pack
cd agent-starter-pack
pip install -e .
```

## Creating an Agent

### Using the CLI

```bash
# Create a new agent for Agent Engine deployment
agent-starter-pack create my-agent -d agent_engine -a adk_base

# Create a RAG agent
agent-starter-pack create rag-agent -d agent_engine -a agentic_rag

# Create for Cloud Run deployment
agent-starter-pack create my-agent -d cloud_run -a adk_base
```

**Options:**
- `-d, --deployment`: `agent_engine` or `cloud_run`
- `-a, --agent-type`: Template to use (see [Agent Templates](#agent-templates))

### Project Structure

```
my-agent/
├── src/
│   ├── agent.py           # Main agent code
│   ├── tools.py           # Tool definitions
│   └── config.py          # Configuration
├── tests/
│   ├── test_agent.py      # Unit tests
│   └── test_integration.py # Integration tests
├── evaluations/
│   └── eval_config.yaml   # Evaluation configuration
├── terraform/             # Infrastructure as code
│   ├── main.tf
│   └── variables.tf
├── .github/
│   └── workflows/
│       ├── test.yaml      # CI pipeline
│       └── deploy.yaml    # CD pipeline
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Agent Templates

### 1. ADK Base Agent (`adk_base`)

Simple agent with function calling using ADK:

```python
# src/agent.py
from google import genai

def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_agent():
    return genai.Agent(
        model="gemini-2.0-flash-exp",
        tools=[get_current_time],
        system_instruction="You are a helpful assistant."
    )
```

**Use cases:**
- Simple question-answering agents
- Agents with a few custom tools
- Starting point for custom agents

### 2. Agentic RAG (`agentic_rag`)

RAG agent using Vertex AI RAG Engine:

```python
# src/agent.py
from google.cloud import aiplatform
from vertexai.preview import rag
from google import genai

def setup_rag_corpus(corpus_name: str, documents: list[str]):
    """Create RAG corpus with documents."""
    corpus = rag.create_corpus(display_name=corpus_name)

    # Import documents
    rag.import_files(
        corpus.name,
        documents,
        chunk_size=512,
        chunk_overlap=50
    )

    return corpus

def create_rag_agent(corpus_name: str):
    # RAG retrieval tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant context from knowledge base."""
        results = rag.retrieval_query(
            corpus_name=corpus_name,
            text=query,
            similarity_top_k=5
        )
        return "\n\n".join([r.text for r in results])

    return genai.Agent(
        model="gemini-2.5-flash",
        tools=[retrieve_context],
        system_instruction="""You are a helpful assistant with access to a knowledge base.
        Use the retrieve_context tool to find relevant information before answering."""
    )
```

**Use cases:**
- Document Q&A systems
- Knowledge base assistants
- Technical support agents

### 3. Multi-Agent System (`multi_agent`)

Orchestrate multiple specialized agents:

```python
# src/agent.py
from google import genai

# Specialized agents
research_agent = genai.Agent(
    model="gemini-2.5-flash",
    tools=[search_web, read_documents],
    system_instruction="You are a research specialist. Gather comprehensive information."
)

analysis_agent = genai.Agent(
    model="gemini-2.5-pro",
    tools=[analyze_data, generate_insights],
    system_instruction="You are an analysis specialist. Provide deep insights."
)

# Supervisor agent
def create_supervisor():
    def delegate_to_researcher(query: str) -> str:
        """Delegate to research agent."""
        return research_agent.query(query).text

    def delegate_to_analyst(data: str) -> str:
        """Delegate to analysis agent."""
        return analysis_agent.query(data).text

    return genai.Agent(
        model="gemini-2.5-flash",
        tools=[delegate_to_researcher, delegate_to_analyst],
        system_instruction="""You are a supervisor. Coordinate between:
        - Research agent: for gathering information
        - Analysis agent: for deep analysis
        Delegate tasks appropriately."""
    )
```

**Use cases:**
- Complex research and analysis workflows
- Systems requiring specialized expertise
- Tasks with distinct phases (research, analysis, synthesis)

### 4. Live API Agent (`live_api`)

Real-time agent with streaming and live data:

```python
# src/agent.py
from google import genai
import asyncio

async def get_live_stock_price(symbol: str) -> float:
    """Get real-time stock price."""
    # Integration with live API
    pass

async def create_live_agent():
    return genai.Agent(
        model="gemini-2.0-flash-exp",
        tools=[get_live_stock_price],
        system_instruction="Provide real-time stock market information.",
        config={"streaming": True}
    )

async def stream_response(query: str):
    agent = await create_live_agent()
    async for chunk in agent.query_stream(query):
        print(chunk.text, end="", flush=True)
```

**Use cases:**
- Real-time data monitoring
- Live chat applications
- Streaming responses for better UX

## Deployment

### Deploy to Agent Engine

```bash
# From your agent directory
cd my-agent

# Set up Google Cloud project
gcloud config set project YOUR_PROJECT_ID

# Deploy using CLI
agent-starter-pack deploy --target agent_engine

# Or use Terraform
cd terraform
terraform init
terraform apply
```

The deployment script:
1. Builds the agent code
2. Packages dependencies
3. Deploys to Vertex AI Agent Engine
4. Configures monitoring and logging
5. Returns the agent endpoint URL

### Deploy to Cloud Run

```bash
# Deploy to Cloud Run for HTTP endpoint
agent-starter-pack deploy --target cloud_run

# This creates a containerized service with:
# - Auto-scaling
# - HTTPS endpoint
# - Authentication
# - Health checks
```

### Manual Deployment

```python
# src/deploy.py
from google.cloud import aiplatform
from google import genai
from agent import create_agent

aiplatform.init(project="your-project-id", location="us-central1")
client = genai.Client(vertexai=True)

agent = create_agent()

remote_agent = client.agent_engines.create(
    agent,
    config={
        "requirements": ["google-cloud-aiplatform[agent_engines,langchain]"],
        "machine_type": "n1-standard-4",
    },
)

print(f"Deployed: {remote_agent.resource_name}")
```

## CI/CD Integration

Agent Starter Pack includes GitHub Actions workflows:

### Testing Pipeline (`.github/workflows/test.yaml`)

```yaml
name: Test Agent

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest tests/ --cov=src

      - name: Run evaluations
        run: agent-starter-pack evaluate
```

### Deployment Pipeline (`.github/workflows/deploy.yaml`)

```yaml
name: Deploy Agent

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Deploy to Agent Engine
        run: |
          agent-starter-pack deploy --target agent_engine
```

## Evaluation

### Evaluation Configuration

```yaml
# evaluations/eval_config.yaml
evaluations:
  - name: "Accuracy Test"
    type: "accuracy"
    test_cases:
      - input: "What is 2 + 2?"
        expected_output: "4"
      - input: "What is the capital of France?"
        expected_output: "Paris"
    threshold: 0.9

  - name: "Tool Usage Test"
    type: "tool_calling"
    test_cases:
      - input: "Search for information about AI"
        expected_tools: ["search_web"]
      - input: "Calculate 15 * 24"
        expected_tools: ["calculator"]
    threshold: 1.0

  - name: "Response Quality"
    type: "llm_judge"
    judge_model: "gemini-2.5-pro"
    criteria:
      - "Relevance"
      - "Accuracy"
      - "Completeness"
    threshold: 0.85
```

### Running Evaluations

```bash
# Run all evaluations
agent-starter-pack evaluate

# Run specific evaluation
agent-starter-pack evaluate --name "Accuracy Test"

# Generate evaluation report
agent-starter-pack evaluate --report
```

### Programmatic Evaluation

```python
# tests/test_integration.py
from agent import create_agent
import pytest

def test_agent_tool_calling():
    agent = create_agent()

    # Test calculator tool
    response = agent.query("What is 15 * 24?")
    assert "360" in response.text

    # Test that tool was called
    assert any(
        tool.name == "calculator"
        for tool in response.tool_calls
    )

def test_agent_accuracy():
    agent = create_agent()

    test_cases = [
        ("What is 2 + 2?", "4"),
        ("What is the capital of France?", "Paris"),
    ]

    for question, expected in test_cases:
        response = agent.query(question)
        assert expected in response.text
```

## Monitoring and Observability

### Built-in Monitoring

Agent Starter Pack automatically configures:

1. **Cloud Logging** - All agent queries and responses
2. **Cloud Monitoring** - Latency, error rate, token usage
3. **Cloud Trace** - Request tracing for debugging
4. **Cost tracking** - Token usage and costs

### Custom Metrics

```python
# src/agent.py
from google.cloud import monitoring_v3
import time

def create_agent_with_monitoring():
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{PROJECT_ID}"

    def track_query(query: str, response: str, latency: float):
        """Track custom metrics."""
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/agent/query_latency"
        series.resource.type = "global"

        point = monitoring_v3.Point()
        point.value.double_value = latency
        point.interval.end_time.seconds = int(time.time())
        series.points = [point]

        client.create_time_series(name=project_name, time_series=[series])

    agent = genai.Agent(...)

    # Wrapper to track metrics
    original_query = agent.query
    def query_with_tracking(q: str):
        start = time.time()
        response = original_query(q)
        latency = time.time() - start
        track_query(q, response.text, latency)
        return response

    agent.query = query_with_tracking
    return agent
```

### Viewing Metrics

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision"

# View metrics in Cloud Console
# Navigate to: Monitoring > Dashboards > Agent Engine
```

### Alerting

```python
# terraform/monitoring.tf
resource "google_monitoring_alert_policy" "agent_error_rate" {
  display_name = "Agent Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "Error rate > 5%"

    condition_threshold {
      filter          = "resource.type=\"agent_engine\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}
```

## Best Practices

1. **Start with templates** - Use built-in templates as starting points
2. **Test locally first** - Run evaluations before deployment
3. **Use CI/CD** - Automate testing and deployment
4. **Monitor in production** - Set up alerts and dashboards
5. **Iterate based on metrics** - Use evaluation results to improve agents
6. **Version control** - Tag releases and track changes
7. **Cost management** - Monitor token usage and set budgets
