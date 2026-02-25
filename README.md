# Retail Saarthi Agent (Sarvam-M + LangGraph)

This project contains a notebook-based prototype for **Retail Saarthi**, an assistant focused on helping Indian kirana and small retailers reduce inventory overhang and improve stock decisions. The notebook sets up a language-aware system prompt, initializes a Sarvam-M-backed chat model, and builds a simple LangGraph loop to simulate agent behavior with inventory and forecasting tools.

It is intended for experimentation and iteration (prompt tuning, message handling, and agent flow testing) before production hardening. **Note:** SARVAM-M currently does not support tool calling in this setup, so tool execution must be updated in the next patch (for example, by switching to a tool-calling-capable model/provider or by adding manual tool-routing logic).

## Prerequisites

- Python 3.11+
- `SARVAM_API_KEY` in your environment (or `.env`)
- Jupyter support in VS Code (or local Jupyter)

## Setup with `uv` (recommended, `pyproject.toml` present)

```bash
# from project root
uv sync

# install dotenv package used in notebook (if not already present)
uv add python-dotenv
```

Run notebook kernel with project environment:

```bash
uv run python -m ipykernel install --user --name rs-slm-agent --display-name "Python (rs-slm-agent)"
```

## If you are not using `uv` (pip fallback)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install \
	ipykernel \
	ipython \
	langchain-core \
	langchain-openai \
	langchain \
	langgraph \
	langgraph-cli[inmem] \
	langsmith \
	python-dotenv
```

## Run

1. Open `Agent-Building.ipynb`.
2. Select the project kernel.
3. Run cells top-to-bottom.
