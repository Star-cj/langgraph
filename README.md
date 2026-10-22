---
title: Star-Grace PDF & Web Q&A
emoji: 📚
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
---

# PDF & Web Question-Answering Agent

A set of Python scripts that answer questions with a LangGraph agent powered by an LLM through OpenRouter. Some versions answer only from your PDF files. Others search your PDFs first and fall back to the internet. Each version has a terminal interface, and some have a Streamlit web interface.

## Features

- **Chat with an LLM** through a LangGraph agent, with conversation memory
- **Single-PDF Q&A**: ask questions about one PDF file
- **Multi-PDF Q&A**: load every PDF in the `pdf/` folder and search across all of them
- **Hybrid search**: check your PDFs first, then search the internet if the answer is not found
- **Confidence scoring**: the hybrid agent returns an answer only at 75% confidence or higher
- **Streamlit UI**: chat interface, sidebar status, batch testing, and a clear-chat button

## Project Structure

```
project/
├── key.py                       # Your OpenRouter API key (local use only, do not upload)
├── requirements.txt             # Packages for Hugging Face Spaces
├── pdf/                         # PDF files used by the multi-PDF and hybrid versions
│   ├── Star-Grace_IT_Consulting.pdf
│   └── Star-Grace_IT_Consulting_Extended.pdf
├── __pycache__/                 # Created automatically by Python (do not upload)
├── main.py                      # Simple chat with the LLM (terminal)
├── mainLit.py                   # Simple chat with the LLM (Streamlit)
├── pdfReader.py                 # Ask questions about one PDF (terminal)
├── pdfsame.py                   # Ask questions about all PDFs in pdf/ (terminal)
├── app.py                       # Ask questions about all PDFs in pdf/ (Streamlit)
├── hybridPDF.py                 # PDFs first, then internet (terminal)
└── hybridPDFlit.py              # PDFs first, then internet (Streamlit)
```

| File | Interface | What it does |
|------|-----------|--------------|
| `main.py` | Terminal | General chat with memory. No PDFs. |
| `mainLit.py` | Streamlit | Same chat as `main.py`, in a web page. |
| `pdfReader.py` | Terminal | Asks for a PDF path, then answers questions about that one file. |
| `pdfsame.py` | Terminal | Loads every PDF in `pdf/`. Includes `batch`, `list`, and `status` commands. |
| `app.py` | Streamlit | Same multi-PDF search as `pdfsame.py`, with a chat UI and a batch test button. This is the Space's main app. |
| `hybridPDF.py` | Terminal | Searches PDFs first. If the answer is not found, searches DuckDuckGo (or Google) and scores the result. |
| `hybridPDFlit.py` | Streamlit | Same hybrid logic as `hybridPDF.py`, with a chat UI. |

The PDFs included in `pdf/`:

- `Star-Grace_IT_Consulting.pdf`
- `Star-Grace_IT_Consulting_Extended.pdf`

## Requirements

- Python 3.10 or newer
- An OpenRouter account with credits: https://openrouter.ai

Install the packages:

```bash
pip install -r requirements.txt
```

## Setup

### Run on your computer

1. **Create the API key file.** In the project folder, create `key.py`:

```python
   OPENROUTER_API_KEY = "your-openrouter-key-here"
```

2. **Add your PDFs** (for the multi-PDF and hybrid versions). Put them in a folder named `pdf` next to the scripts. The project already includes the two Star-Grace IT Consulting PDFs.

3. **Check your credits.** OpenRouter reserves credits for each request based on `max_tokens`. See Troubleshooting below.

### Deploy on Hugging Face Spaces

1. Create a new Space with the **Streamlit** SDK.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `pdf/` folder. Do not upload `key.py` or `__pycache__/`.
3. In the Space, open **Settings → Variables and secrets** and add a secret named `OPENROUTER_API_KEY` with your key as the value.
4. The Space builds and starts automatically. The `app_file` setting at the top of this README tells it to run `app.py`.

## Running the Project

### Terminal versions

```bash
python main.py          # General chat
python pdfReader.py     # Enter a PDF path when prompted
python pdfsame.py       # Reads every PDF in the pdf/ folder
python hybridPDF.py     # PDFs first, then internet
```

Type `exit` to quit. The terminal versions also accept these commands:

- `pdfsame.py`: `batch` (run 10 test questions), `list` (show PDFs), `status` (show folder info)
- `hybridPDF.py`: `test` (run 6 sample questions)

### Streamlit versions (on your computer)

```bash
streamlit run mainLit.py
streamlit run app.py
streamlit run hybridPDFlit.py
```

Streamlit opens the app in your browser at `http://localhost:8501`. Stop the server with `Ctrl+C` in the terminal.

The Streamlit PDF apps (`app.py` and `hybridPDFlit.py`) have these sidebar controls:

- **🔄 Reload PDFs**: rereads the `pdf/` folder after you add or change files
- **🧪 Run batch test**: runs the test questions and shows the results
- **🗑️ Clear chat**: clears the conversation
- **🛑 End session**: clears this browser session's data

## How It Works

### Multi-PDF Q&A (`pdfsame.py` and `app.py`)

1. Every PDF in `pdf/` is read with `pypdf`.
2. The text is split into chunks of about 1,000 words with a 200-word overlap. Each chunk is tagged with its source file name.
3. For a question, the retriever scores each chunk by how many words it shares with the question and keeps the top 5.
4. The LLM receives those chunks and answers only from them. If the answer is not present, it says so.

### Hybrid Q&A (`hybridPDF.py` and `hybridPDFlit.py`)

1. **Step 1: PDFs.** The top PDF chunks are sent to the LLM. An answer is used only if a second LLM call rates its confidence at 75% or higher.
2. **Step 2: Internet.** If the PDFs do not give a confident answer, the question is sent to DuckDuckGo's HTML search (with Google as a fallback). The top results go to the LLM, which returns an answer and a confidence score.
3. **Step 3: No answer.** If neither source gives a confident answer, the agent says it could not find reliable information.

### Single-PDF Q&A (`pdfReader.py`)

Works like the multi-PDF version, but it reads one PDF whose path you enter at startup.

### Simple Chat (`main.py`, `mainLit.py`)

The full conversation history is sent to the LLM on each turn, so the model remembers earlier messages.

## Configuration

| Setting | Where | Default | Purpose |
|---------|-------|---------|---------|
| `model` | every LLM setup | `glm-5.3-flash` | Model name on OpenRouter |
| `temperature` | every LLM setup | `0.1` to `0.7` | Lower values give more consistent answers |
| `max_tokens` | `app.py` | `500` | Limits reply length and reserved credits |
| `chunk_size` / `overlap` | `chunk_text()` | `1000` / `200` | Size of text chunks in words |
| `k` | `get_relevant_documents()` | `5` (or `3`) | Number of chunks sent to the LLM |
| Confidence threshold | hybrid files | `0.75` | Minimum confidence to return an answer |

## Troubleshooting

**`402 ... This request requires more credits, or fewer max_tokens`**
The model's default output limit is very large, and OpenRouter reserves credits for it before the request runs. Add `max_tokens` to the `ChatOpenAI(...)` call, for example `max_tokens=500`. If the error continues, add credits at https://openrouter.ai/settings/credits.

**`Could not find any PDFs` or `Folder 'pdf' not found`**
Check that the `pdf` folder is in the same folder as the script and contains files ending in `.pdf`. The scripts look for the folder relative to their own location, not the current terminal directory.

**`OPENROUTER_API_KEY` is empty or authentication fails on a Space**
Check that the secret is named exactly `OPENROUTER_API_KEY` in the Space settings, then restart the Space.

**Answers say "not found" even though the PDF has the information**
The retriever matches whole words, so questions with different wording may miss. Increase `k`, reduce `chunk_size`, or rephrase the question using the document's terminology.

**PDF text is empty or garbled**
`pypdf` cannot read scanned (image-only) PDFs. Run OCR on them first, for example with `ocrmypdf`.

**Internet search returns nothing**
DuckDuckGo and Google change their HTML often and may block automated requests. Check the CSS class names in `InternetSearch.search_web()` and `_search_fallback()`. Some hosting environments also block these sites.

**`ModuleNotFoundError`**
Install the missing package with `pip install <name>`, or add it to `requirements.txt` for a Space. For `key`, make sure `key.py` is in the same folder as the script when running locally.

**Streamlit does not update after editing a file**
Stop the server with `Ctrl+C` and run `streamlit run <file>.py` again. For changed PDFs, click **🔄 Reload PDFs**.

## Limitations

- Retrieval uses keyword overlap, not embeddings, so it is less accurate on paraphrased questions.
- Internet answers depend on the HTML of search engines, which may change.
- Conversation memory is held in memory only. Restarting the program or Space clears it.
- Confidence scores are produced by the LLM itself, so they are estimates, not guarantees.
- Check answers against the source documents before relying on them for important decisions.

## Security

- **Never upload `key.py`.** It contains your API key. On a Space, store the key as a secret instead.
- **Delete the `__pycache__/` folder before uploading.** Its `.pyc` files can contain compiled copies of your key.
- For a project in Git, add these lines to `.gitignore`:

```
  key.py
  __pycache__/
```

- Do not share your API key in screenshots or error logs. If a key is exposed, revoke it in your OpenRouter settings and create a new one.

## License

Add your license here (for example, MIT).