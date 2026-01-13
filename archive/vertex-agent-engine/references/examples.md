# Complete Agent Examples

This reference provides complete, production-ready agent examples for common use cases.

## Table of Contents

- [Customer Support Agent](#customer-support-agent)
- [Data Analysis Agent](#data-analysis-agent)
- [Code Review Agent](#code-review-agent)
- [Content Generation Agent](#content-generation-agent)
- [Research Assistant Agent](#research-assistant-agent)

## Customer Support Agent

A customer support agent that can search a knowledge base, check order status, and escalate to humans.

### Implementation

```python
from google import genai
from google.cloud import aiplatform
from typing import Literal
import json

# Define tools
def search_knowledge_base(query: str, category: str = "general") -> str:
    """Search the knowledge base for relevant articles.

    Args:
        query: Search query
        category: Category to search in (general, billing, technical, returns)
    """
    # In production, integrate with your knowledge base
    knowledge_base = {
        "billing": {
            "refund policy": "Refunds are available within 30 days...",
            "payment methods": "We accept credit cards, PayPal...",
        },
        "technical": {
            "login issues": "If you can't log in, try resetting your password...",
            "performance": "For performance issues, clear your cache...",
        },
    }

    results = []
    if category in knowledge_base:
        for topic, content in knowledge_base[category].items():
            if query.lower() in topic.lower():
                results.append(f"{topic}: {content}")

    return "\n".join(results) if results else "No relevant articles found."


def check_order_status(order_id: str) -> str:
    """Check the status of an order.

    Args:
        order_id: The order ID to check
    """
    # In production, query your order management system
    orders = {
        "ORD-12345": {
            "status": "shipped",
            "tracking": "TRACK-123",
            "eta": "2025-12-30"
        },
        "ORD-67890": {
            "status": "processing",
            "eta": "2025-12-31"
        },
    }

    if order_id in orders:
        order = orders[order_id]
        return json.dumps(order, indent=2)
    return f"Order {order_id} not found."


def create_support_ticket(
    customer_email: str,
    issue_type: Literal["billing", "technical", "product", "other"],
    description: str,
    priority: Literal["low", "medium", "high"] = "medium"
) -> str:
    """Create a support ticket for complex issues requiring human assistance.

    Args:
        customer_email: Customer's email address
        issue_type: Type of issue
        description: Detailed description of the issue
        priority: Priority level
    """
    # In production, integrate with your ticketing system
    ticket_id = f"TICKET-{hash(customer_email + description) % 100000:05d}"

    return f"""Support ticket created:
    Ticket ID: {ticket_id}
    Type: {issue_type}
    Priority: {priority}
    Email: {customer_email}

    A support agent will contact you within 24 hours."""


# Create the agent
support_agent = genai.Agent(
    model="gemini-2.5-flash",
    tools=[search_knowledge_base, check_order_status, create_support_ticket],
    system_instruction="""You are a helpful customer support agent.

    Guidelines:
    1. Always be polite, empathetic, and professional
    2. Search the knowledge base first for common questions
    3. Use check_order_status for order-related queries
    4. Create a support ticket only when:
       - The issue is complex and requires human intervention
       - You cannot resolve it with available tools
       - The customer explicitly requests to speak with a human
    5. Summarize the resolution clearly
    6. Ask if there's anything else you can help with

    Categories for knowledge base:
    - billing: refunds, payments, invoices
    - technical: login, performance, bugs
    - general: account, shipping, policies
    - returns: return process, warranty""",
    config={
        "temperature": 0.3,  # Lower temperature for consistency
        "max_output_tokens": 512,
    }
)

# Deploy
aiplatform.init(project="your-project-id", location="us-central1")
client = genai.Client(vertexai=True)

remote_agent = client.agent_engines.create(
    support_agent,
    config={
        "requirements": ["google-cloud-aiplatform[agent_engines,langchain]>=1.112"],
        "machine_type": "n1-standard-2",
    },
)

# Test queries
print(remote_agent.query("I can't log into my account").text)
print(remote_agent.query("What's the status of order ORD-12345?").text)
print(remote_agent.query("I need help with a complex billing issue").text)
```

## Data Analysis Agent

An agent that can analyze datasets, generate visualizations, and provide insights.

```python
from google import genai
from google.cloud import aiplatform
import pandas as pd
import json

def analyze_dataset(csv_data: str, analysis_type: str) -> str:
    """Analyze a CSV dataset.

    Args:
        csv_data: CSV data as string
        analysis_type: Type of analysis (summary, correlation, outliers, trends)
    """
    import io

    df = pd.read_csv(io.StringIO(csv_data))

    if analysis_type == "summary":
        return df.describe().to_json()
    elif analysis_type == "correlation":
        return df.corr().to_json()
    elif analysis_type == "outliers":
        # Simple outlier detection using IQR
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
        return outliers.to_json()
    elif analysis_type == "trends":
        # Calculate basic trends
        trends = {}
        for col in df.select_dtypes(include=['number']).columns:
            trends[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        return json.dumps(trends, indent=2)

    return "Analysis type not supported"


def query_data(csv_data: str, query: str) -> str:
    """Query dataset using natural language.

    Args:
        csv_data: CSV data as string
        query: Natural language query (e.g., "rows where sales > 1000")
    """
    import io

    df = pd.read_csv(io.StringIO(csv_data))

    # Simple query parsing (in production, use more robust parsing)
    try:
        if ">" in query:
            col, val = query.replace("rows where", "").split(">")
            col = col.strip()
            val = float(val.strip())
            result = df[df[col] > val]
        elif "average" in query or "mean" in query:
            col = query.replace("average", "").replace("mean", "").strip()
            result = df[col].mean()
            return f"Average {col}: {result}"
        else:
            return "Query not understood. Try: 'rows where column > value' or 'average column'"

        return result.to_json()
    except Exception as e:
        return f"Error executing query: {str(e)}"


def generate_insights(data_summary: str) -> str:
    """Generate business insights from data summary.

    Args:
        data_summary: JSON summary of data analysis
    """
    # Use LLM to generate insights
    insights_prompt = f"""Based on this data summary, provide 3-5 key insights and actionable recommendations:

{data_summary}

Format as:
- Insight 1: [description]
- Insight 2: [description]
...
"""

    # In production, you might call another model or use the agent's model
    return insights_prompt


# Create the agent
data_agent = genai.Agent(
    model="gemini-2.5-pro",  # Use Pro for complex analysis
    tools=[analyze_dataset, query_data, generate_insights],
    system_instruction="""You are a data analysis expert.

    Workflow:
    1. When given a dataset, first run a summary analysis
    2. Look for correlations, outliers, and trends as needed
    3. Use query_data to answer specific questions about the data
    4. Generate insights and recommendations
    5. Explain findings in clear, non-technical language

    Always provide:
    - Clear explanations of what the data shows
    - Visual descriptions (e.g., "The data shows an upward trend...")
    - Actionable recommendations based on the analysis""",
    config={
        "temperature": 0.2,
        "max_output_tokens": 2048,
    }
)

# Deploy
remote_agent = client.agent_engines.create(
    data_agent,
    config={
        "requirements": [
            "google-cloud-aiplatform[agent_engines,langchain]>=1.112",
            "pandas",
            "numpy",
        ],
        "machine_type": "n1-standard-4",
    },
)
```

## Code Review Agent

An agent that reviews code for best practices, security issues, and improvements.

```python
from google import genai
from google.cloud import aiplatform
import re

def check_security_issues(code: str, language: str) -> str:
    """Check for common security vulnerabilities.

    Args:
        code: Source code to check
        language: Programming language (python, javascript, java, etc.)
    """
    issues = []

    # Python security checks
    if language.lower() == "python":
        if "eval(" in code:
            issues.append("CRITICAL: Using eval() can lead to code injection")
        if "exec(" in code:
            issues.append("CRITICAL: Using exec() can lead to code injection")
        if "pickle.load" in code and "untrusted" in code.lower():
            issues.append("HIGH: Unpickling untrusted data can lead to RCE")
        if re.search(r'password\s*=\s*["\']', code):
            issues.append("HIGH: Hardcoded password detected")
        if "shell=True" in code:
            issues.append("MEDIUM: shell=True in subprocess can be dangerous")

    # JavaScript security checks
    elif language.lower() == "javascript":
        if "eval(" in code:
            issues.append("CRITICAL: Using eval() can lead to code injection")
        if "innerHTML" in code and "user" in code.lower():
            issues.append("HIGH: Setting innerHTML with user input can cause XSS")
        if "document.write" in code:
            issues.append("MEDIUM: document.write can be problematic")

    if not issues:
        return "No obvious security issues found."

    return "Security issues found:\n" + "\n".join(f"- {issue}" for issue in issues)


def check_code_quality(code: str, language: str) -> str:
    """Check code quality and best practices.

    Args:
        code: Source code to check
        language: Programming language
    """
    suggestions = []

    lines = code.split("\n")

    # Check line length
    long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 100]
    if long_lines:
        suggestions.append(f"Lines too long (>100 chars): {long_lines[:5]}")

    # Check for TODOs
    todos = [i for i, line in enumerate(lines, 1) if "TODO" in line or "FIXME" in line]
    if todos:
        suggestions.append(f"Unresolved TODOs/FIXMEs at lines: {todos}")

    # Language-specific checks
    if language.lower() == "python":
        if "except:" in code:
            suggestions.append("Avoid bare except clauses - catch specific exceptions")
        if not re.search(r'""".*?"""', code, re.DOTALL) and "def " in code:
            suggestions.append("Add docstrings to functions")

    if not suggestions:
        return "Code quality looks good!"

    return "Quality suggestions:\n" + "\n".join(f"- {s}" for s in suggestions)


def suggest_improvements(code: str, language: str) -> str:
    """Suggest code improvements and optimizations.

    Args:
        code: Source code to analyze
        language: Programming language
    """
    improvements = []

    # Python improvements
    if language.lower() == "python":
        if "for i in range(len(" in code:
            improvements.append("Consider using enumerate() instead of range(len())")
        if re.search(r'\+ str\(', code):
            improvements.append("Consider using f-strings for string formatting")
        if "if x == True" in code or "if x == False" in code:
            improvements.append("Use 'if x:' or 'if not x:' instead of comparing to True/False")

    # General improvements
    if code.count("\n") > 50:
        improvements.append("Consider breaking this into smaller functions")

    if not improvements:
        return "No obvious improvements needed."

    return "Suggested improvements:\n" + "\n".join(f"- {i}" for i in improvements)


# Create the agent
code_review_agent = genai.Agent(
    model="gemini-2.5-pro",
    tools=[check_security_issues, check_code_quality, suggest_improvements],
    system_instruction="""You are an expert code reviewer.

    Review process:
    1. Check for security vulnerabilities first
    2. Review code quality and adherence to best practices
    3. Suggest specific improvements with examples
    4. Prioritize issues: CRITICAL > HIGH > MEDIUM > LOW
    5. Provide explanations for why each issue matters
    6. Offer code examples for suggested fixes

    Format your review as:
    ## Security Review
    [findings]

    ## Code Quality
    [findings]

    ## Suggested Improvements
    [findings with code examples]

    ## Summary
    [overall assessment and priority actions]

    Be constructive and educational in your feedback.""",
    config={
        "temperature": 0.3,
        "max_output_tokens": 2048,
    }
)

# Deploy
remote_agent = client.agent_engines.create(
    code_review_agent,
    config={
        "requirements": ["google-cloud-aiplatform[agent_engines,langchain]>=1.112"],
        "machine_type": "n1-standard-2",
    },
)
```

## Content Generation Agent

An agent that generates blog posts, social media content, and marketing copy.

```python
from google import genai
from google.cloud import aiplatform
from typing import Literal

def research_topic(topic: str, sources: int = 3) -> str:
    """Research a topic to gather information.

    Args:
        topic: Topic to research
        sources: Number of sources to consult
    """
    # In production, integrate with search APIs or knowledge bases
    research_data = f"""Research on '{topic}':
    - Key points about {topic}
    - Current trends and statistics
    - Expert opinions and insights
    - Related topics and context"""

    return research_data


def check_seo_keywords(content: str, target_keywords: list[str]) -> str:
    """Check if content includes target SEO keywords.

    Args:
        content: Content to check
        target_keywords: List of keywords to check for
    """
    results = {}
    content_lower = content.lower()

    for keyword in target_keywords:
        count = content_lower.count(keyword.lower())
        results[keyword] = {
            "count": count,
            "density": round(count / len(content.split()) * 100, 2) if content else 0
        }

    return str(results)


def optimize_for_platform(
    content: str,
    platform: Literal["blog", "twitter", "linkedin", "instagram", "facebook"]
) -> str:
    """Optimize content for specific platform.

    Args:
        content: Original content
        platform: Target platform
    """
    optimizations = {
        "twitter": {
            "max_length": 280,
            "style": "concise, hashtags, engaging hook",
            "format": "Single tweet or thread"
        },
        "linkedin": {
            "max_length": 3000,
            "style": "professional, thought leadership, data-driven",
            "format": "Well-structured with line breaks"
        },
        "instagram": {
            "max_length": 2200,
            "style": "visual-focused, storytelling, emotive",
            "format": "Caption with emojis and hashtags"
        },
        "facebook": {
            "max_length": 5000,
            "style": "conversational, community-building",
            "format": "Engaging narrative with call-to-action"
        },
        "blog": {
            "max_length": None,
            "style": "in-depth, SEO-optimized, informative",
            "format": "Structured with headers, lists, examples"
        }
    }

    guidelines = optimizations.get(platform, optimizations["blog"])

    return f"""Optimization guidelines for {platform}:
    {guidelines}

    Original content length: {len(content)} characters
    Recommended: {guidelines['max_length'] or 'No limit'}
    Style: {guidelines['style']}
    Format: {guidelines['format']}"""


# Create the agent
content_agent = genai.Agent(
    model="gemini-2.5-flash",
    tools=[research_topic, check_seo_keywords, optimize_for_platform],
    system_instruction="""You are an expert content creator and copywriter.

    Content creation process:
    1. Research the topic thoroughly using research_topic
    2. Create engaging, original content tailored to the audience
    3. Check SEO keyword inclusion if provided
    4. Optimize for the target platform
    5. Include clear call-to-action when appropriate

    Guidelines:
    - Write in a clear, engaging style appropriate for the platform
    - Use storytelling techniques to maintain interest
    - Include specific examples and data when possible
    - Optimize for readability (short paragraphs, bullet points)
    - Ensure content is original and valuable

    For blog posts:
    - Use headers (##, ###) for structure
    - Include introduction, body, conclusion
    - Add relevant examples and case studies

    For social media:
    - Hook readers in the first line
    - Use appropriate hashtags
    - Include call-to-action
    - Match platform's tone and style""",
    config={
        "temperature": 0.8,  # Higher creativity for content generation
        "max_output_tokens": 2048,
    }
)

# Deploy
remote_agent = client.agent_engines.create(
    content_agent,
    config={
        "requirements": ["google-cloud-aiplatform[agent_engines,langchain]>=1.112"],
        "machine_type": "n1-standard-2",
    },
)

# Example usage
blog_post = remote_agent.query("""Write a blog post about AI agents in customer service.
Target keywords: AI agents, customer service automation, chatbots
Length: 800 words""")

linkedin_post = remote_agent.query("""Create a LinkedIn post about the importance of AI ethics.
Platform: linkedin
Tone: professional thought leadership""")
```

## Research Assistant Agent

An agent that can search, summarize, and synthesize information from multiple sources.

```python
from google import genai
from google.cloud import aiplatform

def search_papers(query: str, field: str = "computer science") -> str:
    """Search academic papers and publications.

    Args:
        query: Search query
        field: Academic field (computer science, medicine, physics, etc.)
    """
    # In production, integrate with arXiv, Google Scholar, PubMed, etc.
    papers = f"""Found 5 papers on '{query}' in {field}:
    1. "Paper Title 1" (2024) - Summary of key findings...
    2. "Paper Title 2" (2023) - Summary of key findings...
    3. "Paper Title 3" (2024) - Summary of key findings..."""

    return papers


def summarize_document(document_text: str, length: str = "medium") -> str:
    """Summarize a long document.

    Args:
        document_text: Full text of document
        length: Summary length (short, medium, long)
    """
    # Use the LLM to summarize
    lengths = {
        "short": 100,
        "medium": 300,
        "long": 500
    }

    max_words = lengths.get(length, 300)

    summary_prompt = f"""Summarize the following document in approximately {max_words} words:

{document_text[:10000]}  # Limit input length

Focus on:
- Main arguments and findings
- Key data and evidence
- Conclusions and implications"""

    return summary_prompt


def compare_sources(source1: str, source2: str, aspect: str = "general") -> str:
    """Compare two sources on a specific aspect.

    Args:
        source1: First source text
        source2: Second source text
        aspect: Aspect to compare (methodology, findings, conclusions, etc.)
    """
    comparison_prompt = f"""Compare these two sources on '{aspect}':

Source 1:
{source1[:2000]}

Source 2:
{source2[:2000]}

Provide:
- Similarities
- Differences
- Strengths and weaknesses of each
- Which source is more credible/comprehensive"""

    return comparison_prompt


def synthesize_information(sources: list[str], research_question: str) -> str:
    """Synthesize information from multiple sources.

    Args:
        sources: List of source texts
        research_question: The research question to answer
    """
    synthesis_prompt = f"""Research question: {research_question}

Based on these {len(sources)} sources, provide a comprehensive synthesis:

Sources:
{chr(10).join(f'{i+1}. {s[:500]}...' for i, s in enumerate(sources))}

Include:
- Overall consensus or disagreements
- Key themes and patterns
- Gaps in current research
- Implications and recommendations"""

    return synthesis_prompt


# Create the agent
research_agent = genai.Agent(
    model="gemini-2.5-pro",  # Use Pro for research tasks
    tools=[search_papers, summarize_document, compare_sources, synthesize_information],
    system_instruction="""You are an expert research assistant.

    Research process:
    1. Understand the research question or topic
    2. Search for relevant academic papers and sources
    3. Summarize key documents
    4. Compare and contrast different sources
    5. Synthesize findings into coherent analysis
    6. Identify gaps and future research directions

    Guidelines:
    - Always cite sources (though citations are simulated in this example)
    - Acknowledge limitations and uncertainties
    - Present multiple perspectives when they exist
    - Distinguish between correlation and causation
    - Highlight contradictory findings
    - Provide evidence-based conclusions

    Output format:
    ## Research Summary
    [Overview of topic and findings]

    ## Key Findings
    - Finding 1 (Source)
    - Finding 2 (Source)

    ## Analysis
    [Synthesis and comparison]

    ## Gaps and Future Research
    [What's missing]

    ## Conclusion
    [Evidence-based conclusion]""",
    config={
        "temperature": 0.3,
        "max_output_tokens": 4096,
    }
)

# Deploy
remote_agent = client.agent_engines.create(
    research_agent,
    config={
        "requirements": ["google-cloud-aiplatform[agent_engines,langchain]>=1.112"],
        "machine_type": "n1-standard-4",  # More resources for complex tasks
    },
)

# Example usage
research_output = remote_agent.query("""Research the current state of AI agents in healthcare.
Focus on:
- Clinical decision support
- Patient monitoring
- Drug discovery
- Ethical considerations

Provide a comprehensive synthesis with key findings and future directions.""")

print(research_output.text)
```

## Best Practices Across All Examples

1. **Error Handling**: Wrap tool functions in try-except blocks
2. **Input Validation**: Validate inputs before processing
3. **Rate Limiting**: Implement rate limiting for external API calls
4. **Logging**: Log all tool calls and results for debugging
5. **Testing**: Write unit tests for all tool functions
6. **Documentation**: Provide clear docstrings for all tools
7. **Security**: Never expose sensitive credentials in code
8. **Monitoring**: Track tool usage and performance metrics
