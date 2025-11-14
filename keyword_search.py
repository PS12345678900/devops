from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

from .data_loader import Chunk, load_from_base_dir
from .vector_store import RetrievedChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _build_query_terms(query: str) -> List[str]:
    # Simple: split into alphanumeric tokens
    return _tokenize(query)


def _score_chunk(query_terms: List[str], chunk: Chunk) -> float:
    if not chunk.text:
        return 0.0
    tokens = _tokenize(chunk.text)
    if not tokens:
        return 0.0
    term_set = set(tokens)
    score = 0.0
    # Term presence score
    for qt in query_terms:
        if qt in term_set:
            score += 1.0
    # Frequency boost
    freq = {t: 0 for t in query_terms}
    for t in tokens:
        if t in freq:
            freq[t] += 1
    score += 0.2 * sum(min(3, c) for c in freq.values())
    # Metadata boosts
    md = chunk.metadata or {}
    title = str(md.get("title", "") or md.get("id", ""))
    section = str(md.get("section", ""))
    tags = md.get("tags", [])
    services = md.get("services", [])
    meta_text = f"{title} {section} {' '.join(map(str, tags or []))} {' '.join(map(str, services or []))}"
    meta_tokens = set(_tokenize(meta_text))
    for qt in query_terms:
        if qt in meta_tokens:
            score += 0.5
    # Section prioritization
    section_l = section.lower()
    if section_l == "remediation":
        score *= 1.2
    elif section_l == "diagnosis":
        score *= 1.1
    # Length normalization to avoid overly long chunks dominating
    score = score / (1.0 + math.log(1 + len(tokens)))
    return score


def keyword_retrieve(base_dir: str, query: str, top_k: int = 10) -> List[RetrievedChunk]:
    chunks = load_from_base_dir(base_dir)
    if not chunks:
        return []
    q_terms = _build_query_terms(query)
    scored: List[Tuple[float, Chunk]] = []
    for ch in chunks:
        s = _score_chunk(q_terms, ch)
        if s > 0.0:
            scored.append((s, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    results: List[RetrievedChunk] = []
    for s, ch in top:
        results.append(RetrievedChunk(document=ch.text, metadata=ch.metadata, score=float(s)))
    return results


