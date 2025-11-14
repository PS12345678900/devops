import os
import sys
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure project root is on sys.path so `import app.*` works when running the script directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# Minimal .env loader (no extra dependency): loads KEY=VALUE pairs into environment
def _load_dotenv_from_root() -> None:
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f.read().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Don't override existing explicit environment
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        # Fail silently; users can still set env vars via shell
        pass


_load_dotenv_from_root()

# Defensive: clear cert envs that can break SSL context on some Windows setups
for _v in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
    if os.environ.get(_v):
        try:
            os.environ.pop(_v, None)
        except Exception:
            pass

from app.generation import synthesize_checklist_rule_based, synthesize_checklist_with_llm
from app.keyword_search import keyword_retrieve


st.set_page_config(page_title="Incident Response Assistant", layout="wide")
st.title("Incident Response Assistant")
st.caption("Ask questions grounded in your uploaded/bundled playbooks and runbooks (keyword search only).")


# No vector store / embeddings in keyword-only mode


def sidebar_controls() -> Dict[str, Any]:
    st.sidebar.header("Settings")
    top_k = st.sidebar.number_input("Top K", min_value=5, max_value=50, value=20, step=1)
    use_llm = st.sidebar.checkbox("Use OpenAI for synthesis", value=False)
    st.sidebar.markdown("---")
    st.sidebar.caption("Optional: set OPENAI_API_KEY for LLM synthesis (formatting).")
    return {
        "top_k": int(top_k),
        "use_llm": use_llm,
    }


def upsert_controls(cfg: Dict[str, Any]) -> None:
    st.subheader("Optional: Upsert New Data")
    st.caption("Upload files to include in search (stored under staging_uploads/).")
    uploaded = st.file_uploader("Upload files (YAML/MD/LOG/TXT)", type=["yaml","yml","md","log","txt"], accept_multiple_files=True)
    do_upload = st.button("Save Uploaded Files")

    if do_upload and uploaded:
        staging = os.path.join(_PROJECT_ROOT, "staging_uploads")
        os.makedirs(staging, exist_ok=True)
        pb_dir = os.path.join(staging, "playbooks"); os.makedirs(pb_dir, exist_ok=True)
        rb_dir = os.path.join(staging, "runbooks"); os.makedirs(rb_dir, exist_ok=True)
        lg_dir = os.path.join(staging, "logs"); os.makedirs(lg_dir, exist_ok=True)
        for f in uploaded:
            name = os.path.basename(f.name)
            ext = os.path.splitext(name)[1].lower()
            if ext in [".yaml", ".yml"]:
                target = os.path.join(pb_dir, name)
            elif ext == ".md":
                target = os.path.join(rb_dir, name)
            else:
                target = os.path.join(lg_dir, name)
            with open(target, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to staging_uploads/. They are now searchable.")


def retrieval_and_guidance_ui(cfg: Dict[str, Any]) -> None:
    st.subheader("Ask")
    prompt = st.text_area("Your prompt", value="payments-api experiencing 5xx spike after deploy. What should I do?")
    do_ask = st.button("Ask", type="primary")
    show_sources = st.checkbox("Show sources", value=True)

    if not do_ask:
        return

    with st.spinner("Retrieving…"):
        # Prefer uploaded files; else fallback to bundled data
        base_dir = os.path.join(_PROJECT_ROOT, "staging_uploads")
        if not os.path.isdir(base_dir):
            base_dir = os.path.join(_PROJECT_ROOT, "data")
        retrieved = keyword_retrieve(base_dir=base_dir, query=prompt, top_k=cfg["top_k"])

    st.caption(f"Retrieved {len(retrieved)} chunks")
    if not retrieved:
        st.warning("No results found. Upload files (or place them under data/) and try again.")
        return

    if cfg["use_llm"]:
        items = synthesize_checklist_with_llm(prompt, retrieved, severity="P2")
    else:
        items = synthesize_checklist_rule_based(prompt, retrieved, severity="P2")

    st.markdown("#### Guided steps")
    md_lines: List[str] = []
    for i, it in enumerate(items):
        title = it.get("label") or (it.get("section", "") or "Action").title()
        st.markdown(f"**Step {i+1}. {title}**")
        details = it.get("details", "").strip()
        if details and details != title:
            st.write(details)
        if it.get("command"):
            st.write("Command:")
            st.code(it["command"], language="bash")
        if it.get("verify"):
            st.write(f"Verify: {it['verify']}")
        if it.get("rollback"):
            st.write(f"Rollback: {it['rollback']}")
        if show_sources and it.get("ref"):
            st.caption(f"Ref: {it['ref']}")
        st.markdown("---")
        # Build export markdown
        md_lines.append(f"- [ ] {title}")
        if details and details != title:
            md_lines.append(f"  - Context: {details.splitlines()[0]}")
        if it.get("command"):
            md_lines.append(f"  - Command: `{it['command']}`")
        if it.get("verify"):
            md_lines.append(f"  - Verify: {it['verify']}")
        if it.get("rollback"):
            md_lines.append(f"  - Rollback: {it['rollback']}")
        if show_sources and it.get("ref"):
            md_lines.append(f"  - Ref: {it['ref']}")
    md_text = "\n".join(md_lines)
    st.download_button("Download Markdown", data=md_text, file_name="incident_checklist.md", mime="text/markdown")


def main() -> None:
    cfg = sidebar_controls()
    st.markdown("Upload files if needed, then ask a question. Search is keyword-based only.")

    upsert_controls(cfg)
    retrieval_and_guidance_ui(cfg)


if __name__ == "__main__":
    main()


