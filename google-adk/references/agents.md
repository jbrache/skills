# Agents in ADK

Complete guide to agent types, multi-agent systems, workflow patterns, and instruction engineering.

## Table of Contents
- [Agent Types](#agent-types)
- [Multi-Agent Systems](#multi-agent-systems)
- [Workflow Agents](#workflow-agents)
- [Agent Instruction Engineering](#agent-instruction-engineering)
- [Coordination Patterns](#coordination-patterns)
- [Production Examples](#production-examples)

---

## Agent Types

### LLM Agent

The primary agent type for most use cases. Uses a language model to process requests and coordinate tool usage.

```python
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="assistant",
    model="gemini-3-flash-preview",
    instruction="You are a helpful assistant...",
    description="Brief description for coordination",
    tools=[...]  # Optional tools
)
```

See [Configuration → Model Selection](configuration.md#model-selection) for model options.

### Custom Agent (BaseAgent)

For implementing custom logic without LLM inference.

```python
from google.adk.agents import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name=name, description="...")

    def execute(self, input_data, context):
        # Custom logic here
        return result
```

### Agent Properties

| Property | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Unique identifier for the agent |
| `model` | Yes | Gemini model to use |
| `instruction` | Yes | System prompt defining agent behavior |
| `description` | No | Brief description for multi-agent coordination |
| `tools` | No | List of tools available to the agent |
| `sub_agents` | No | Child agents for hierarchical systems |
| `response_model` | No | Pydantic schema for structured output |
| `callbacks` | No | Callback handlers for observability |

---

## Multi-Agent Systems

### When to Use Multi-Agent vs Single Agent

| Scenario | Single | Multi | Reasoning |
|----------|:------:|:-----:|-----------|
| Simple Q&A with tools | ✅ | ❌ | No coordination overhead needed |
| Different expertise domains | ❌ | ✅ | Specialists with focused instructions |
| Different model requirements | ❌ | ✅ | Use Pro for reasoning, Flash for speed |
| Parallel research/analysis | ❌ | ✅ | Concurrent execution saves time |
| Iterative refinement | ❌ | ✅ | Creator → Reviewer → Refiner pattern |
| Single task, multiple tools | ✅ | ❌ | One agent can use many tools |
| Rapid prototyping | ✅ | ❌ | Start simple, add agents as needed |

**Start single, scale to multi-agent when:**
1. Instructions become too complex for one agent
2. You need different models for different subtasks
3. You want independent testing/deployment of components
4. Parallel execution would significantly improve performance

### Hierarchical Agent Structure

```python
from google.adk.agents import LlmAgent

# Define specialized agents
greeter = LlmAgent(
    name="greeter",
    model="gemini-3-flash-preview",
    instruction="Greet users warmly and professionally.",
    description="Handles user greetings"
)

task_executor = LlmAgent(
    name="task_executor",
    model="gemini-3-flash-preview",
    instruction="Execute tasks efficiently.",
    description="Executes specific tasks"
)

# Create coordinator with sub-agents
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-3-flash-preview",
    instruction="Coordinate between greeting and task execution.",
    description="Routes requests to specialized agents",
    sub_agents=[greeter, task_executor]
)
```

**How it works:**
- ADK engine and model guide agents to work together
- Coordinator delegates to appropriate sub-agents
- Each agent maintains its own context
- Full conversation history shared across hierarchy

---

## Workflow Agents

Specialized agents for structured workflows.

### Sequential Agent

Execute agents in a defined sequence. Use when steps must run in order.

```python
from google.adk.agents.workflow import SequentialAgent

# Research → Analyze → Report pipeline
research_pipeline = SequentialAgent(
    name="research_pipeline",
    agents=[researcher, analyzer, writer]
)
```

**Best for:** Data pipelines, Research → Analysis → Report workflows.

### Parallel Agent

Execute multiple agents concurrently. Use for independent tasks.

```python
from google.adk.agents.workflow import ParallelAgent

# Gather from multiple sources simultaneously
parallel_research = ParallelAgent(
    name="parallel_research",
    agents=[news_agent, academic_agent, social_agent]
)
```

**Best for:** Gathering data from multiple sources, running parallel analyses.

### Loop Agent

Execute agents in a loop with conditions. Use for iterative refinement.

```python
from google.adk.agents.workflow import LoopAgent

def should_continue(context) -> bool:
    return not context.get("test_passed", False) and context.get("iteration", 0) < 5

iterative_coder = LoopAgent(
    name="iterative_coder",
    agent=code_and_test_agent,
    condition=should_continue,
    max_iterations=5
)
```

**Best for:** Iterative refinement, retry with modifications, quality thresholds.

### Combining Workflows

```python
# Parallel first, then synthesize
comprehensive_research = SequentialAgent(
    name="comprehensive_research",
    agents=[
        parallel_research,  # Fan-out: gather in parallel
        synthesizer         # Combine results
    ]
)
```

---

## Agent Instruction Engineering

### Principles

Good agent instructions are:
1. **Specific**: Clear role, responsibilities, and boundaries
2. **Action-oriented**: Describe what to do, not just what the agent is
3. **Tool-aware**: Guide proper tool usage without micromanaging
4. **Contextual**: Include relevant domain knowledge and constraints
5. **Delegation-ready**: (For coordinators) Explain when to delegate

### Basic Agent Pattern

**Bad:**
```python
instruction="You are a helpful assistant."
```

**Good:**
```python
instruction="""You are a weather information specialist.

Your role:
- Answer questions about current weather and forecasts
- Use get_weather tool for real-time data
- Provide temperature in Celsius unless user specifies Fahrenheit
- Include relevant details: temperature, conditions, humidity, wind

When you don't know:
- Ask for clarification if location is ambiguous
- Admit when data is unavailable rather than guessing
"""
```

### Research Agent Pattern

```python
instruction="""You are a research analyst specializing in technical topics.

Research process:
1. Use google_search to find authoritative sources
2. Verify information across multiple sources
3. Prioritize recent information (last 2 years) for technology topics
4. Cite sources with URLs in your summary

Quality standards:
- Distinguish between facts, opinions, and speculation
- Note conflicting information when found
- Highlight confidence level (high/medium/low) for key findings

Output format:
- Start with executive summary (2-3 sentences)
- Provide detailed findings with source citations
- End with "Confidence: [level]" and any caveats
"""
```

### Coordinator Agent Pattern

```python
instruction="""You are an intelligent coordinator managing specialized agents.

Available agents:
- researcher: Finds and analyzes information
- task_executor: Executes code and system operations
- writer: Creates formatted documents and reports

Delegation strategy:
1. Analyze the user request to identify required capabilities
2. Delegate to the most appropriate specialist agent
3. Coordinate multiple agents for complex multi-step tasks

Your role:
- Route requests to appropriate specialists
- Maintain context across agent interactions
- Synthesize results from multiple agents when needed
- Handle ambiguous requests by asking clarifying questions first

Do NOT:
- Attempt tasks yourself that specialists should handle
- Delegate without understanding the request
- Over-delegate simple tasks that don't require specialist expertise
"""
```

---

## Coordination Patterns

### Pattern 1: Router Coordinator

Simple delegation based on request type.

```python
router_coordinator = LlmAgent(
    name="router",
    model="gemini-3-flash-preview",  # Fast routing
    instruction="""Analyze user requests and delegate to appropriate specialist:
    - Questions about data, statistics → data_analyst
    - Code-related tasks → software_engineer
    - Writing, documentation → technical_writer

    For ambiguous requests, ask clarifying questions first.""",
    sub_agents=[data_analyst, software_engineer, technical_writer]
)
```

### Pattern 2: Orchestrator Coordinator

Manages complex multi-step processes across agents.

```python
orchestrator = LlmAgent(
    name="project_orchestrator",
    model="gemini-3-pro-preview",  # Complex coordination
    instruction="""Manage software development workflow:

    1. Start with requirements_analyst to clarify scope
    2. Delegate to architect for design decisions
    3. Assign implementation to developer
    4. Send to qa_engineer for testing
    5. If QA finds issues, return to developer
    6. Once tests pass, delegate to documenter

    Quality gates:
    - Don't proceed to implementation without clear requirements
    - Don't mark complete until tests pass""",
    sub_agents=[requirements_analyst, architect, developer, qa_engineer, documenter]
)
```

### Pattern 3: Consensus Coordinator

Multiple agents analyze same problem, coordinator synthesizes.

```python
# Three analysts with different approaches
conservative = LlmAgent(name="conservative", instruction="Analyze with risk-averse approach...")
innovative = LlmAgent(name="innovative", instruction="Analyze with forward-thinking approach...")
pragmatic = LlmAgent(name="pragmatic", instruction="Analyze with practical, balanced approach...")

consensus_coordinator = LlmAgent(
    name="consensus",
    model="gemini-3-pro-preview",
    instruction="""Coordinate analysis from three perspectives:
    1. Present problem to all analysts
    2. Identify areas of agreement (strong signals)
    3. Identify disagreements (risks/tradeoffs)
    4. Synthesize balanced recommendation""",
    sub_agents=[conservative, innovative, pragmatic]
)
```

### Pattern 4: Feedback Loop

Iterative refinement with quality checking.

```python
creator = LlmAgent(name="creator", instruction="Create initial draft...")
reviewer = LlmAgent(name="reviewer", instruction="Review for accuracy, clarity, style...")
refiner = LlmAgent(name="refiner", instruction="Improve based on feedback...")

feedback_coordinator = LlmAgent(
    name="feedback_loop",
    instruction="""Coordinate iterative refinement:
    1. Send to creator for initial draft
    2. Send to reviewer for evaluation
    3. If approved: return final content
    4. If not: send draft + feedback to refiner
    5. Repeat (max 3 iterations)""",
    sub_agents=[creator, reviewer, refiner]
)
```

---

## Production Examples

### Customer Support System

```python
from google.adk.agents import LlmAgent
from google.adk.sessions import Session

# Tier 1: FAQ and simple issues
tier1 = LlmAgent(
    name="tier1_support",
    model="gemini-3-flash-preview",
    instruction="""You are Tier 1 customer support.

    Your role:
    - Answer common questions using knowledge base
    - Provide step-by-step guidance for basic troubleshooting

    Escalate to Tier 2 if:
    - Issue requires account access or data modification
    - Technical issue beyond basic troubleshooting
    - Customer tried KB solutions without success""",
    description="Handles common questions and basic troubleshooting",
    tools=[search_knowledge_base]
)

# Tier 2: Technical issues
tier2 = LlmAgent(
    name="tier2_support",
    model="gemini-3-pro-preview",
    instruction="""You are Tier 2 technical support.

    Your expertise:
    - Complex technical troubleshooting
    - API and integration issues
    - Performance and configuration problems

    Create tickets for:
    - Bugs requiring engineering investigation
    - Issues needing database modifications""",
    description="Handles complex technical issues",
    tools=[search_knowledge_base, google_search, create_support_ticket]
)

# Coordinator
support_coordinator = LlmAgent(
    name="support_coordinator",
    model="gemini-3-flash-preview",
    instruction="""Route support requests:
    - Start with tier1_support
    - Escalate to tier2_support when tier1 cannot resolve""",
    sub_agents=[tier1, tier2]
)

# Usage
def handle_request(user_id: str, message: str):
    session = Session(session_id=f"support-{user_id}")
    return support_coordinator.run(input=message, session=session)
```

### Research Pipeline

```python
from google.adk.agents import LlmAgent
from google.adk.agents.workflow import SequentialAgent, ParallelAgent
from pydantic import BaseModel

class ResearchReport(BaseModel):
    title: str
    summary: str
    findings: list[str]
    sources: list[str]

# Parallel research phase
parallel_research = ParallelAgent(
    name="parallel_research",
    agents=[
        LlmAgent(name="news", instruction="Research recent news...", tools=[google_search]),
        LlmAgent(name="academic", instruction="Find academic sources...", tools=[google_search]),
        LlmAgent(name="trends", instruction="Identify market trends...", tools=[google_search])
    ]
)

# Analysis and synthesis
analyzer = LlmAgent(
    name="analyzer",
    model="gemini-3-pro-preview",
    instruction="Synthesize findings, identify patterns...",
)

# Report writer
writer = LlmAgent(
    name="writer",
    model="gemini-3-pro-preview",
    instruction="Create executive report...",
    response_model=ResearchReport
)

# Complete pipeline
research_pipeline = SequentialAgent(
    name="research_pipeline",
    agents=[parallel_research, analyzer, writer]
)

# Usage
result = research_pipeline.run(input="Research the AI customer service market")
report = result.parsed  # ResearchReport instance
```

---

## Best Practices

### Agent Design
1. **Agent granularity**: Create focused agents with clear responsibilities
2. **Descriptions**: Write clear descriptions for effective coordination
3. **Tool scope**: Assign tools only to agents that need them
4. **Testing**: Test agents individually before composing into systems

### Workflow Design
1. **Match pattern to need**: Sequential for dependencies, parallel for independence
2. **Consider coordination cost**: More agents = more coordination overhead
3. **Plan for failure**: Include error handling and retry logic
4. **Optimize model choice**: Flash for simple tasks, Pro for complex reasoning

### Instructions
1. **Be specific**: Define clear role, responsibilities, boundaries
2. **Include process**: Explain how to approach tasks
3. **Guide tool usage**: Mention tools and when to use them
4. **Handle edge cases**: Specify what to do when uncertain
5. **Set quality bars**: Define what "good" output looks like

## Documentation References

- Custom agents: https://github.com/google/adk-docs/blob/main/docs/agents/custom-agents.md
- LLM agents: https://github.com/google/adk-docs/blob/main/docs/agents/llm-agents.md
- Multi-agents: https://github.com/google/adk-docs/blob/main/docs/agents/multi-agents.md
- Workflow agents: https://github.com/google/adk-docs/blob/main/docs/agents/workflow-agents/index.md
- Models: https://github.com/google/adk-docs/blob/main/docs/agents/models.md
