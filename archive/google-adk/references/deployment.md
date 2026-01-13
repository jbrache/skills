# Deploying ADK Agents

ADK agents can be deployed to various Google Cloud platforms for production use.

## Local Development

Test agents locally before deploying to production:

```bash
# Start interactive web interface
uv run adk web

# Opens at http://localhost:8080
```

The web interface provides interactive testing with chat UI, tool execution visualization, and session management.

For more local development options and patterns, see [Project Structure - Local Development](project-structure.md#local-development).

---

## Deployment Options

### Vertex AI Agent Engine (Recommended)

Vertex AI Agent Engine provides managed infrastructure optimized for AI agents.

**Prerequisites:**
- Google Cloud project with billing enabled
- Vertex AI API enabled
- ADK CLI installed

**Steps:**

1. **Prepare agent configuration**

```python
# agent.py
from google.adk.agents import LlmAgent

root_agent = LlmAgent(
    name="vertex_assistant",
    model="gemini-3-flash-preview",
    instruction="You are a helpful assistant.",
    tools=[...]
)
```

2. **Deploy to Vertex AI**

```bash
# Using ADK CLI (recommended)
adk deploy \
    --agent-module agent:root_agent \
    --platform vertex-ai \
    --project my-project \
    --region us-central1

# Or using gcloud
gcloud ai agents deploy \
    --agent-file=agent.yaml \
    --region=us-central1
```

**Benefits:**
- Managed scaling and infrastructure
- Integrated monitoring and logging
- Built-in security features
- Optimized for Gemini models
- Session management and memory
- Auto-scaling based on demand

**Configuration options:**
- `--region`: Deployment region (default: us-central1)
- `--min-instances`: Minimum instances (default: 0)
- `--max-instances`: Maximum instances for auto-scaling
- `--timeout`: Request timeout in seconds

### Cloud Run

Alternative serverless deployment with full control over the server implementation.

**Prerequisites:**
- Google Cloud project with billing enabled
- Cloud Run API enabled
- Docker installed locally

**Steps:**

1. **Create a Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Expose port
EXPOSE 8080

# Run the agent server
CMD ["python", "main.py"]
```

2. **Create main.py with server**

```python
from google.adk.agents import LlmAgent
from google.adk.runtime import serve
import os

# Define your agent
agent = LlmAgent(
    name="my_assistant",
    model="gemini-3-flash-preview",
    instruction="You are a helpful assistant.",
    tools=[...]
)

# Serve on Cloud Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    serve(agent, host="0.0.0.0", port=port)
```

3. **Deploy to Cloud Run**

```bash
# Build and deploy in one command
gcloud run deploy my-agent \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars GOOGLE_CLOUD_PROJECT=my-project

# Or build Docker image first
docker build -t gcr.io/my-project/my-agent .
docker push gcr.io/my-project/my-agent

gcloud run deploy my-agent \
    --image gcr.io/my-project/my-agent \
    --region us-central1 \
    --allow-unauthenticated
```

**Configuration options:**
- `--memory`: Set memory limit (default: 512Mi)
- `--cpu`: Set CPU allocation (default: 1)
- `--timeout`: Request timeout (default: 300s, max: 3600s)
- `--max-instances`: Maximum concurrent instances
- `--min-instances`: Minimum instances (for faster cold starts)

**Best for:** Custom server implementations, containerized deployments, full control over infrastructure

## Configuration

### Environment Variables

```python
import os

# Common configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = os.environ.get("MODEL", "gemini-3-flash-preview")

agent = LlmAgent(
    name="configurable_assistant",
    model=MODEL,
    instruction="You are a helpful assistant."
)
```

### Runtime Configuration

```python
from google.adk.runtime import RunConfig

config = RunConfig(
    max_turns=10,
    timeout=30,
    streaming=True,
    enable_observability=True
)

serve(agent, config=config, port=8080)
```

## API Endpoints

Deployed agents expose REST API endpoints:

### POST /chat

Send messages to the agent:

```bash
curl -X POST https://my-agent-url/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, agent!",
    "session_id": "user-123"
  }'
```

### POST /chat/stream

Stream responses:

```bash
curl -X POST https://my-agent-url/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me a story",
    "session_id": "user-123"
  }'
```

### GET /health

Health check endpoint:

```bash
curl https://my-agent-url/health
```

## Security

### Authentication

Enable authentication on Cloud Run:

```bash
gcloud run deploy my-agent \
    --source . \
    --region us-central1 \
    --no-allow-unauthenticated

# Clients must include auth token
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    https://my-agent-url/chat
```

### Service Account

Use service accounts for Google Cloud API access:

```bash
# Create service account
gcloud iam service-accounts create adk-agent \
    --display-name "ADK Agent"

# Grant permissions
gcloud projects add-iam-policy-binding my-project \
    --member "serviceAccount:adk-agent@my-project.iam.gserviceaccount.com" \
    --role "roles/aiplatform.user"

# Deploy with service account
gcloud run deploy my-agent \
    --source . \
    --service-account adk-agent@my-project.iam.gserviceaccount.com
```

### Secret Management

Use Secret Manager for sensitive data:

```python
from google.cloud import secretmanager

def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

API_KEY = get_secret("api-key")
```

## Monitoring

### Cloud Logging

```python
import logging
from google.cloud import logging as cloud_logging

# Setup Cloud Logging
client = cloud_logging.Client()
client.setup_logging()

# Log from your agent
logger = logging.getLogger(__name__)
logger.info("Agent started")
logger.error("Tool execution failed", extra={"tool": "search"})
```

### Cloud Monitoring

Set up alerts and dashboards:

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=my-agent"

# Create alert
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="Agent Error Rate" \
    --condition-threshold-value=0.1
```

## Scaling

### Auto-scaling Configuration

Cloud Run auto-scales based on traffic:

```bash
gcloud run deploy my-agent \
    --source . \
    --min-instances 1 \
    --max-instances 100 \
    --concurrency 80
```

### Cost Optimization

- Use `--min-instances 0` for infrequent traffic
- Use `--min-instances 1+` to avoid cold starts
- Choose appropriate CPU/memory allocation
- Use `gemini-3-flash-preview` for cost-effective deployments
- Implement caching for repeated queries

## Testing Deployments

```bash
# Local testing
uv run adk web

# Or run server directly
python agent.py

# Test deployed endpoint
curl -X POST https://my-agent-url/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# Load testing
ab -n 1000 -c 10 -p request.json \
  -T "application/json" \
  https://my-agent-url/chat
```

## Documentation References

- Vertex AI Agent Engine: https://github.com/google/adk-docs/blob/main/docs/deploy/agent-engine.md
- Cloud Run: https://github.com/google/adk-docs/blob/main/docs/deploy/cloud-run.md
- Runtime config: https://github.com/google/adk-docs/blob/main/docs/runtime/runconfig.md
