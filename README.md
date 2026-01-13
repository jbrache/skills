# Skills
Skills are folders of instructions, scripts, and resources that Claude loads dynamically to improve performance on specialized tasks. Skills teach Claude how to complete specific tasks in a repeatable way, whether that's creating documents with your company's brand guidelines, analyzing data using your organization's specific workflows, or automating personal tasks.

For more information, check out:
- [What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills)
- [Using skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [How to create custom skills](https://support.claude.com/en/articles/12512198-creating-custom-skills)
- [Equipping agents for the real world with Agent Skills](https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)

# About This Repository

The Google Agent Development Kit (ADK) skill streamlines the entire lifecycle of building ADK applications, from initial scaffolding to verification against best practices. It helps you quickly start new projects with the latest SDK versions and ensures your applications follow official documentation patterns.

## How It Works

This skills use the [MCP LLMS-TXT Doc Server](https://pypi.org/project/mcpdoc/) to provide the Claude model with up-to-date information about ADK. The documentation content is sourced from the `llms.txt` file in the [official ADK-Docs](https://github.com/google/adk-docs). This file is effectively a sitemap of the ADK documentation.

Each skills `SKILL.md` file provides instructions to the Claude model, guiding it to use the tools provided by this MCP Server when answering questions about ADK.

## Why Is This Useful?

This extension empowers Claude CLI to provide accurate and current information about ADK, without relying on potentially outdated internal knowledge. This is particularly useful for:

*   Answering questions about ADK's features and APIs.
*   Assisting with development tasks related to ADK.
*   Ensuring that the information provided by the Claude CLI is consistent with the latest ADK documentation.

## Disclaimer

**These skills are provided for demonstration and educational purposes only.** While some of these capabilities may be available in Claude, the implementations and behaviors you receive from Claude may differ from what is shown in these skills. These skills are meant to illustrate patterns and possibilities. Always test skills thoroughly in your own environment before relying on them for critical tasks.

# Skill Sets
- [./skills](./skills): Skill examples for creating and verifying Google Agent Development Kit (ADK) agents

# Try in Claude Code, Claude.ai, and the API

```bash
# Set your PROJECT_ID
export PROJECT_ID=<project id>

# Clone Locally if you want to add ADK skill to Claude Code
git clone https://github.com/jbrache/skills.git

# Setup application default credentials
gcloud auth application-default login
gcloud config set project $PROJECT_ID

# Set up environment variables for Claude:
export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=global
export ANTHROPIC_VERTEX_PROJECT_ID=$PROJECT_ID

# Set up environment variables for Vertex AI:
export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
```

To add an MCP server for the [ADK docs to Claude Code](https://google.github.io/adk-docs/tutorials/coding-with-ai/#claude-code), run the following command:
```bash
claude mcp add adk-docs --scope user --transport stdio -- uvx --from mcpdoc mcpdoc --urls AgentDevelopmentKit:https://google.github.io/adk-docs/llms.txt --transport stdio
```

## Claude Code
You can register this repository as a Claude Code Plugin marketplace by running the following command in Claude Code:
```
/plugin marketplace add jbrache/skills

# Remove When Done
/plugin marketplace remove jbrache-agent-skills
```

Then, to install a specific set of skills:
1. Select `Browse and install plugins`
2. Select `jbrache-agent-skills`
3. Select `google-adk-skills`
4. Select `Install now`

Restart Claude Code to apply changes.

Alternatively, directly [install locally](https://code.claude.com/docs/en/discover-plugins#add-from-local-paths):
```
/plugin marketplace add ./skills
/plugin install google-adk-skills@skills

# Remove When Done
/plugin marketplace remove jbrache-agent-skills
```

After installing the plugin, you can use the ADK skill by just mentioning it. For instance, you can ask Claude Code to do something like: 
```
Use the google-adk-skills:google-adk-dev skill to build portfolio news agent - first get portfolio from the tool. For now just fixed GOOGL and NVDA (pick number of stock) 2.execute search to find news from last week 3. create one paragraph summary of impact on the portfolio
```

## Claude.ai

To use any skill from this repository or upload custom skills, follow the instructions in [Using skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude#h_a4222fa77b).

## Claude API

You can use Anthropic's pre-built skills, and upload custom skills, via the Claude API. See the [Skills API Quickstart](https://docs.claude.com/en/api/skills-guide#creating-a-skill) for more.

## 📝 License

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Disclaimer

This repository itself is not an officially supported Google product. The code in this repository is for demonstrative purposes only.