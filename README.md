## Incident Response Assistant (MVP)

Minimal Streamlit app for checklist-style, referenced guidance from incident playbooks/runbooks/logs. Uses ChromaDB for persistence and OpenAI embeddings. Optional OpenAI synthesis for checklist wording.

### Windows setup (PowerShell, Python 3.11/3.12) — No Anaconda
1) Change to your project directory:
```powershell
cd "C:\Users\Karan Singh-GGN\OneDrive - McKinsey & Company\Documents\cursor\devops"
```

2) Create and activate a virtual environment:
```powershell
# If you have python3.12 on PATH, use: python3.12 -m venv .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4) Set your OpenAI key (required for embeddings):
```powershell
$env:OPENAI_API_KEY="sk-..."
```

5) (Optional) Generate synthetic data (playbooks/runbooks/logs):
```powershell
python .\scripts\generate_synthetic_data.py
```

6) Launch the Streamlit app:
```powershell
streamlit run .\app\streamlit_app.py
```

In the app (left sidebar):
- Collection name (Chroma)
- Upload files to upsert
- Ask questions in the main prompt box; answers are grounded in the indexed data

### One-liner (PowerShell)
```powershell
cd "C:\Users\Karan Singh-GGN\OneDrive - McKinsey & Company\Documents\cursor\devops" ; python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; python -m pip install --upgrade pip ; pip install -r requirements.txt ; $env:OPENAI_API_KEY="sk-..." ; python .\scripts\generate_synthetic_data.py ; streamlit run .\app\streamlit_app.py
```

### Project layout
- `app/`
  - `streamlit_app.py` — UI and flows (ChromaDB only, OpenAI embeddings)
  - `embedding.py` — OpenAI embeddings
  - `data_loader.py` — load and chunk YAML/MD/logs
  - `indexing.py` — build index (embed + upsert)
  - `vector_store.py` — Chroma store implementation
  - `generation.py` — checklist synthesis (rule-based fallback or OpenAI)
- `scripts/generate_synthetic_data.py` — seeds realistic playbooks, runbooks, logs
- `data/` — synthetic data (created by script)
- `.chroma/` — Chroma persistence (auto-created)

### Workflow
1. Upload files and click “Upsert Uploaded Files” (first time only).
2. Ask a question; get a prioritized checklist with references.
3. Export the checklist as Markdown.


