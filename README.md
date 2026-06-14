# LLM Portfolio

A portfolio showcasing LLM engineering projects including competitive intelligence analysis, multi-agent systems, RAG applications, and autonomous AI solutions built with Ollama, OpenAI, and modern LLM techniques.

## Projects

### Competitive Intelligence Analyzer

A multi-agent system that analyzes company websites to generate comprehensive competitive intelligence reports.

**Features:**
- Multi-agent pipeline with specialized AI agents
- Structured JSON communication between agents
- Streaming responses for real-time progress
- Flexible model selection (any Ollama model)
- Robust web scraping (BeautifulSoup + Selenium fallback)
- Agent orchestration with final agent coordinating the analysis pipeline

**Agent Pipeline:**
1. **Business Extractor**: Identifies company information, products, positioning
2. **SWOT Analyzer**: Performs comprehensive SWOT analysis
3. **Competitive Positioning**: Analyzes market position and differentiation
4. **Strategic Advisor**: Orchestrates previous agents and generates actionable recommendations

**Technical Implementation:**
- Web scraping with BeautifulSoup and Selenium
- Multi-step LLM calls with intermediate reasoning
- Document intelligence and business analysis
- Structured outputs and JSON handling
- Streaming responses for real-time feedback
- Model flexibility and configuration
- Error handling and fallback mechanisms
- Agent orchestration patterns

## Setup

1. **Install dependencies:**
```bash
uv sync
```

2. **Configure Ollama:**
```bash
# Install Ollama from https://ollama.com
# Pull your preferred model (e.g., gemma4:e4b)
ollama pull gemma4:e4b
```

3. **Set environment variables:**
```bash
cp .env.example .env
# Edit .env with your preferred model
```

4. **Start Jupyter:**
```bash
uv run jupyter notebook
```

5. **Open the notebook:**
Open `competitive_intelligence_analyzer.ipynb` in Jupyter and follow the examples.

## Usage

Run the Jupyter notebook:
```bash
uv run jupyter notebook
```

Open `competitive_intelligence_analyzer.ipynb` and follow the examples.

## Tech Stack

- **LLM**: Ollama (gemma4:e4b, llama3.2, etc.)
- **Web Scraping**: BeautifulSoup, Selenium
- **API**: OpenAI Python Client (Ollama-compatible)
- **Environment**: Python 3.12+, uv package manager

## Business Applications

This system provides actionable competitive intelligence for:
- Market analysis and positioning
- Strategic planning and decision support
- Investment due diligence
- Partnership evaluation
- Competitive landscape assessment

## License

MIT License
