from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .vector_store import RetrievedChunk

_LLM_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    import httpx  # type: ignore

    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False


def _priority_rank(md: Dict[str, Any]) -> int:
    p = str(md.get("priority", "")).lower()
    if p in ("critical", "urgent", "high", "p1"):
        return 0
    if p in ("medium", "p2"):
        return 1
    if p in ("low", "p3"):
        return 2
    # Default: remediation before diagnosis, then others
    section = str(md.get("section", "")).lower()
    if section == "remediation":
        return 1
    if section == "diagnosis":
        return 2
    return 3


def _format_reference(md: Dict[str, Any]) -> str:
    src = str(md.get("source_path", ""))
    title = str(md.get("title", "") or md.get("id", ""))
    section = str(md.get("section", ""))
    return f"{title or os.path.basename(src)} › {section} ({os.path.basename(src)})"


def synthesize_checklist_rule_based(
    query: str,
    retrieved: List[RetrievedChunk],
    severity: str = "P2",
    max_items: int = 10,
) -> List[Dict[str, str]]:
    # Prefer remediation items, sorted by priority; include diagnosis at the end
    items: List[Dict[str, str]] = []
    retrieved_sorted = sorted(retrieved, key=lambda r: (_priority_rank(r.metadata), -r.score))
    for r in retrieved_sorted:
        section = str(r.metadata.get("section", "")).lower()
        text = r.document.strip()
        ref = _format_reference(r.metadata)
        # Extract lightweight fields if present in text
        command = ""
        verify = ""
        rollback = ""
        lines = text.splitlines()
        for line in lines:
            if line.lower().startswith("command:"):
                command = line.split(":", 1)[1].strip()
            elif line.lower().startswith("verify:"):
                verify = line.split(":", 1)[1].strip()
            elif line.lower().startswith("rollback:"):
                rollback = line.split(":", 1)[1].strip()
        # Derive a human-friendly step label from the first non-field line
        non_field_lines = [
            ln for ln in lines if not ln.lower().startswith(("command:", "verify:", "rollback:"))
        ]
        first_line = non_field_lines[0].strip() if non_field_lines else ""
        label = first_line or (f"{section.title()} step" if section else "Action")
        details = "\n".join(non_field_lines).strip()
        item = {
            "label": label,
            "command": command,
            "verify": verify,
            "rollback": rollback,
            "ref": ref,
            "section": section,
            "details": details,
        }
        items.append(item)
        if len(items) >= max_items:
            break
    # Deduplicate by label
    seen = set()
    deduped: List[Dict[str, str]] = []
    for it in items:
        if it["label"] in seen:
            continue
        deduped.append(it)
        seen.add(it["label"])
    return deduped[:max_items]


def synthesize_checklist_with_llm(
    query: str,
    retrieved: List[RetrievedChunk],
    severity: str = "P2",
    model: Optional[str] = None,
    max_items: int = 10,
) -> List[Dict[str, str]]:
    if not _LLM_AVAILABLE:
        return synthesize_checklist_rule_based(query, retrieved, severity=severity, max_items=max_items)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return synthesize_checklist_rule_based(query, retrieved, severity=severity, max_items=max_items)
    raw_base = os.getenv("OPENAI_BASE_URL", "")
    base_url = (raw_base or "").strip().strip('"').strip("'")
    if not base_url or " " in base_url or not base_url.lower().startswith(("http://", "https://")):
        base_url = None
    model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
    http_client = httpx.Client(verify=False)
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model, http_client=http_client)
    context_blocks = []
    for r in retrieved[:15]:
        ref = _format_reference(r.metadata)
        context_blocks.append(f"[{ref}]\n{r.document}")
    context_text = "\n\n---\n\n".join(context_blocks)
    prompt = (
        "You are an SRE incident assistant. Produce a prioritized, actionable checklist for remediation and diagnosis.\n"
        f"Severity: {severity}\n"
        f"Query: {query}\n\n"
        "Use only the provided context. Cite the reference for each step in parentheses.\n"
        "Each item should be one line for the step, followed by optional indented sub-lines for Command, Verify, Rollback.\n"
        "Limit to {max_items} items.\n\n"
        f"Context:\n{context_text}"
    )
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", "") or ""
    except Exception:
        return synthesize_checklist_rule_based(query, retrieved, severity=severity, max_items=max_items)
    # Simple parse: convert bullet-like lines into structured items
    items: List[Dict[str, str]] = []
    cur: Dict[str, str] = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("- ") or raw.startswith("* "):
            if cur:
                items.append(cur)
            cur = {"label": raw[2:].strip(), "command": "", "verify": "", "rollback": "", "ref": "", "section": ""}
        elif raw.lower().startswith("command:"):
            cur["command"] = raw.split(":", 1)[1].strip()
        elif raw.lower().startswith("verify:"):
            cur["verify"] = raw.split(":", 1)[1].strip()
        elif raw.lower().startswith("rollback:"):
            cur["rollback"] = raw.split(":", 1)[1].strip()
        elif raw.startswith("(") and raw.endswith(")"):
            cur["ref"] = raw.strip("()")
    if cur:
        items.append(cur)
    # Truncate
    items = items[:max_items]
    return items


