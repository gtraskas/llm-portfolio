# LLM Portfolio

A portfolio showcasing LLM engineering projects including competitive intelligence analysis, multi-agent systems, tool calling, RAG applications, and autonomous AI solutions built with Ollama, OpenAI, or any OpenAI-compatible API and modern LLM techniques.

## Projects

This portfolio contains multiple Jupyter notebooks demonstrating different LLM engineering patterns and applications:

### Competitive Intelligence Analyzer

**File:** `competitive_intelligence_analyzer.ipynb`

A multi-agent system that analyzes company websites to generate comprehensive competitive intelligence reports.

### Multi-Agent Chat System

**File:** `multi_agent_chat.ipynb`

A 3-way conversation system with AI agents having distinct personalities and different model configurations.

### Car Rental Customer Service Assistant

**File:** `car_rental_assistant.ipynb`

A customer service assistant for a car rental company with tool calling capabilities and database integration.

## Setup

```bash
# Install dependencies
uv sync

# Configure Ollama (install from https://ollama.com)
ollama pull gemma4:e4b
ollama pull gemma4:e2b
ollama pull llama3.2:3b

# Set environment variables
cp .env.example .env

# Start Jupyter
uv run jupyter notebook
```

## Usage

Run the Jupyter notebook server and open any notebook:

```bash
uv run jupyter notebook
```
