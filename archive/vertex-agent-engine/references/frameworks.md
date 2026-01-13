# Framework Integration

Vertex AI Agent Engine supports multiple agent frameworks beyond ADK, allowing you to deploy agents built with LangChain or LangGraph.

## Table of Contents

- [LangChain Integration](#langchain-integration)
- [LangGraph Integration](#langgraph-integration)
- [Framework Comparison](#framework-comparison)

## LangChain Integration

### Installation

```bash
pip install google-cloud-aiplatform[agent_engines,langchain]>=1.112
pip install langchain langchain-google-vertexai
```

### Creating a LangChain Agent

```python
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# Define tools
def calculator(query: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(query))
    except:
        return "Error in calculation"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for math calculations. Input should be a valid Python expression."
    ),
    Tool(
        name="Search",
        func=search_web,
        description="Useful for searching information on the web."
    ),
]

# Create LLM
llm = ChatVertexAI(model_name="gemini-2.0-flash-exp", temperature=0.7)

# Create prompt template
prompt = PromptTemplate.from_template("""Answer the following question using the available tools.

You have access to the following tools:
{tools}

Use this format:
Question: the input question
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Question: {input}
{agent_scratchpad}""")

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### Deploying LangChain Agents

```python
from google.cloud import aiplatform
from google import genai

aiplatform.init(project="your-project-id", location="us-central1")
client = genai.Client(vertexai=True)

# Deploy the agent executor
remote_agent = client.agent_engines.create(
    agent_executor,
    config={
        "requirements": [
            "google-cloud-aiplatform[agent_engines,langchain]>=1.112",
            "langchain",
            "langchain-google-vertexai",
        ],
    },
)

# Query the deployed agent
response = remote_agent.query("What is 15 * 24 + 36?")
print(response.text)
```

### LangChain with RAG

```python
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and split documents
documents = [...]  # Your documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
vectorstore = FAISS.from_documents(splits, embeddings)

# Create RAG chain
llm = ChatVertexAI(model_name="gemini-2.5-flash")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# Deploy
remote_agent = client.agent_engines.create(
    qa_chain,
    config={
        "requirements": [
            "google-cloud-aiplatform[agent_engines,langchain]>=1.112",
            "langchain",
            "langchain-google-vertexai",
            "faiss-cpu",
        ],
    },
)
```

## LangGraph Integration

### Installation

```bash
pip install google-cloud-aiplatform[agent_engines,langgraph]>=1.112
pip install langgraph langchain-google-vertexai
```

### Creating a LangGraph Agent

LangGraph enables building multi-agent workflows with cycles and state management:

```python
from langgraph.graph import Graph, END
from langchain_google_vertexai import ChatVertexAI
from typing import TypedDict, Annotated
import operator

# Define state
class State(TypedDict):
    messages: Annotated[list, operator.add]
    current_step: str

# Define nodes
def research_node(state: State) -> State:
    """Research information."""
    llm = ChatVertexAI(model_name="gemini-2.5-flash")
    response = llm.invoke(f"Research: {state['messages'][-1]}")
    return {
        "messages": [response.content],
        "current_step": "research_complete"
    }

def analysis_node(state: State) -> State:
    """Analyze researched information."""
    llm = ChatVertexAI(model_name="gemini-2.5-pro")
    response = llm.invoke(f"Analyze: {state['messages'][-1]}")
    return {
        "messages": [response.content],
        "current_step": "analysis_complete"
    }

def synthesis_node(state: State) -> State:
    """Synthesize final answer."""
    llm = ChatVertexAI(model_name="gemini-2.5-flash")
    response = llm.invoke(f"Synthesize: {state['messages'][-1]}")
    return {
        "messages": [response.content],
        "current_step": "complete"
    }

# Build graph
workflow = Graph()

workflow.add_node("research", research_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("synthesis", synthesis_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "analysis")
workflow.add_edge("analysis", "synthesis")
workflow.add_edge("synthesis", END)

app = workflow.compile()
```

### Deploying LangGraph Agents

```python
from google.cloud import aiplatform
from google import genai

aiplatform.init(project="your-project-id", location="us-central1")
client = genai.Client(vertexai=True)

remote_agent = client.agent_engines.create(
    app,
    config={
        "requirements": [
            "google-cloud-aiplatform[agent_engines,langgraph]>=1.112",
            "langgraph",
            "langchain-google-vertexai",
        ],
    },
)

# Query with initial state
response = remote_agent.query(
    "Analyze the impact of AI on healthcare",
    initial_state={"messages": [], "current_step": "start"}
)
print(response.text)
```

### Multi-Agent LangGraph Pattern

```python
from langgraph.graph import Graph

def supervisor_node(state: State) -> State:
    """Supervisor decides which agent to call."""
    # Logic to route to appropriate agent
    pass

def specialist_agent_1(state: State) -> State:
    """Specialist for specific tasks."""
    pass

def specialist_agent_2(state: State) -> State:
    """Another specialist."""
    pass

workflow = Graph()
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("specialist1", specialist_agent_1)
workflow.add_node("specialist2", specialist_agent_2)

# Conditional routing based on supervisor decision
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "specialist1": "specialist1",
        "specialist2": "specialist2",
        "end": END
    }
)
workflow.add_edge("specialist1", "supervisor")
workflow.add_edge("specialist2", "supervisor")

app = workflow.compile()
```

## Framework Comparison

| Feature | ADK | LangChain | LangGraph |
|---------|-----|-----------|-----------|
| **Learning curve** | Low | Medium | Medium-High |
| **Flexibility** | High | High | Very High |
| **Multi-agent** | Yes | Limited | Yes |
| **State management** | Built-in | Manual | Built-in |
| **Tool calling** | Native | Yes | Yes |
| **Code execution** | Via Agent Engine | Via Agent Engine | Via Agent Engine |
| **RAG support** | Yes | Excellent | Excellent |
| **Best for** | Simple agents | RAG, chains | Complex workflows |

### Choosing a Framework

**Use ADK when:**
- Building straightforward agents with tools
- Want tight Google Cloud integration
- Prefer simple, code-first approach
- Don't need complex orchestration

**Use LangChain when:**
- Building RAG applications
- Need extensive integrations (100+ tools)
- Want mature ecosystem and documentation
- Working with vector stores and embeddings

**Use LangGraph when:**
- Building complex, stateful workflows
- Need cyclic graph execution
- Orchestrating multiple specialized agents
- Require fine-grained control over agent flow

**Note on Code Execution:**
All frameworks can use Agent Engine's Code Execution service for running agent-generated code in a secure sandbox. See [agent-engine-services.md](agent-engine-services.md#code-execution) for details.
