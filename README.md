# Self-Correcting RAG & Web Fallback Assistant

A small question-answering app that checks its local company-policy knowledge
base first. If an LLM grader decides the retrieved documents do not answer the
question, the workflow searches DuckDuckGo and uses those results instead.

## Architecture

```text
Question
   |
   v
retrieve_node (Qdrant + all-MiniLM-L6-v2)
   |
   v
grade_node (LLM: enough evidence?)
   | yes                       | no
   v                           v
generate_node              web_search_node (DDGS; Wikipedia API if no results)
   ^                           |
   |                           v
   +---------------------- generate_node
                               |
                               v
                    Answer + source attribution
```

Qdrant stores its local database under `qdrant_data/`; no Docker service is
needed. The sample knowledge base contains eight short AI and company policy
documents. The embedding model is downloaded by Sentence-Transformers the first
time ingestion runs.

## Requirements

- Python 3.10 or newer
- An API key for an OpenAI-compatible chat-completion endpoint (the defaults
  use Groq with the `openai/gpt-oss-20b` model)
- Internet access for the first model download and live web fallback

## Setup (Windows PowerShell)

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Configure the LLM API:

   ```powershell
   Copy-Item .env.example .env
   ```

   Edit `.env` and replace `your_api_key_here` with your provider's key. For
   another OpenAI-compatible provider, set `LLM_BASE_URL` and `LLM_MODEL` to the
   provider's documented endpoint and model. The app also checks `GROQ_API_KEY`,
   `OPENAI_API_KEY`, and `GEMINI_API_KEY` if `LLM_API_KEY` is not set.

4. Build the local knowledge base:

   ```powershell
   python ingest.py
   ```

   Run ingestion again after changing the sample documents.

## Run the apps

Start the FastAPI service in the first terminal:

```powershell
uvicorn main:app --reload
```

Send a question:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"Can I paste customer data into a public AI tool?"}'
```

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.
The service also provides `GET /health`.

In another terminal, launch the Streamlit interface. It sends questions to the
API, so the Qdrant disk database is opened by only one process:

```powershell
streamlit run app.py
```

The UI labels answers as **Local Docs** or **Web Search** and shows the
document titles or web-result links used by the workflow.

## Workflow details

- `retrieve_node` embeds the question and retrieves up to four local documents.
- `grade_node` asks the configured LLM whether those documents are sufficient.
- Only a `"no"` grade routes to `web_search_node`. It searches DuckDuckGo
  through the current `ddgs` package and falls back to Wikipedia when that
  search returns no usable results.
- `generate_node` answers from the selected evidence and appends the source
  attribution. If web results are used, it is prompted to include their URLs.
- The API and UI both report the selected source. An LLM API key is needed for
  grading and answer generation; DuckDuckGo does not require one.
