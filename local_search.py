#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Personal Search Engine (API-free, single-file, no external deps)

Features:
- Recursive indexer for text-ish files (.txt, .md, .py, .json, .csv, .log)
- Inverted index with BM25 ranking (k1, b configurable)
- Incremental indexing (based on file mtime + size)
- Simple query language:
    +term        -> required term
    -term        -> forbidden term
    "some words" -> required phrase
    ext:md       -> restrict results to files with extension .md (may repeat)
  Plain terms are optional (boost score).
- Snippet extraction with highlighting
- CLI + REPL

Usage:
    python local_search.py index /path/to/folder
    python local_search.py search "your query here" --top 10
    python local_search.py show <doc_id>
    python local_search.py repl

Notes:
- No external libraries. PDF support omitted by design to stay API-free and dependency-free.
- Extend FILE_EXTS below to include more types you know are text.
"""

import os
import re
import sys
import math
import time
import json
import pickle
import argparse
import unicodedata
from typing import Dict, List, Tuple, Set, Optional

# -------------------------
# Config
# -------------------------

INDEX_DIR = ".local_search_index"
POSTINGS_FILE = os.path.join(INDEX_DIR, "postings.pkl")
DOCS_FILE = os.path.join(INDEX_DIR, "docs.pkl")
META_FILE = os.path.join(INDEX_DIR, "meta.json")

# File types considered textual by default
FILE_EXTS = {".txt", ".md", ".py", ".json", ".csv", ".log"}

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Minimal English stopwords (extend if you like)
STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is",
    "it","its","of","on","that","the","to","was","were","will","with","or","not"
}

# Token pattern: words with letters/numbers, apostrophes allowed in the middle
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")

HIGHLIGHT_LEFT = "[["
HIGHLIGHT_RIGHT = "]]"

# -------------------------
# Utilities
# -------------------------

def normalize_text(s: str) -> str:
    # Lowercase + NFKC normalization
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    tokens = []
    for m in TOKEN_RE.finditer(s):
        tok = m.group(0)
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens

def read_text_file(path: str) -> Optional[str]:
    # Try utf-8, then fallback to latin-1, ignoring errors as last resort
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            try:
                with open(path, "r", errors="ignore") as f:
                    return f.read()
            except Exception:
                return None

def file_should_index(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in FILE_EXTS

def ensure_index_dir() -> None:
    if not os.path.isdir(INDEX_DIR):
        os.makedirs(INDEX_DIR, exist_ok=True)

def load_pickle(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# -------------------------
# Index data structures
# -------------------------

class Index:
    """
    postings: dict term -> list of (doc_id, term_freq)
    doc_len: dict doc_id -> int (# of tokens)
    doc_path: dict doc_id -> str
    doc_meta: dict doc_id -> {"mtime": float, "size": int}
    avg_doc_len: float
    """
    def __init__(self):
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_len: Dict[int, int] = {}
        self.doc_path: Dict[int, str] = {}
        self.doc_meta: Dict[int, Dict[str, float]] = {}
        self.avg_doc_len: float = 0.0

    def save(self):
        ensure_index_dir()
        save_pickle(POSTINGS_FILE, self.postings)
        save_pickle(DOCS_FILE, (self.doc_len, self.doc_path, self.doc_meta, self.avg_doc_len))
        save_json(META_FILE, {"total_docs": len(self.doc_len), "last_updated": time.time()})

    @staticmethod
    def load() -> "Index":
        idx = Index()
        idx.postings = load_pickle(POSTINGS_FILE, {})
        loaded = load_pickle(DOCS_FILE, None)
        if loaded is None:
            idx.doc_len = {}
            idx.doc_path = {}
            idx.doc_meta = {}
            idx.avg_doc_len = 0.0
        else:
            idx.doc_len, idx.doc_path, idx.doc_meta, idx.avg_doc_len = loaded
        return idx

# -------------------------
# Indexer
# -------------------------

def index_directory(root: str) -> None:
    """
    Build or update the index incrementally.
    """
    print("Loading existing index (if any)...")
    idx = Index.load()

    # Map path -> doc_id for quick lookup
    path_to_id: Dict[str, int] = {}
    for did, p in idx.doc_path.items():
        path_to_id[p] = did

    to_index_paths: List[str] = []
    seen_paths: Set[str] = set()

    print("Scanning files...")
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs like .git
        if os.path.basename(dirpath).startswith("."):
            # Allow index dir itself but skip hidden generally
            pass
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not file_should_index(path):
                continue
            seen_paths.add(os.path.abspath(path))
            st = None
            try:
                st = os.stat(path)
            except Exception:
                continue
            size = int(st.st_size)
            mtime = float(st.st_mtime)

            did = path_to_id.get(path)
            if did is not None:
                meta = idx.doc_meta.get(did)
                if meta is not None:
                    # unchanged?
                    if int(meta.get("size", -1)) == size and float(meta.get("mtime", -1.0)) == mtime:
                        continue
            to_index_paths.append(path)

    # Remove deleted docs from index
    to_delete: List[int] = []
    for did, path in idx.doc_path.items():
        if os.path.abspath(path) not in seen_paths:
            to_delete.append(did)

    if len(to_delete) > 0:
        print("Removing", len(to_delete), "deleted docs from index...")
        # For each term's postings, remove entries with those doc_ids
        delete_set = set(to_delete)
        # We avoid list comprehensions by manual rebuild
        new_postings = {}
        for term, plist in idx.postings.items():
            new_list = []
            i = 0
            while i < len(plist):
                (doc_id, tf) = plist[i]
                if doc_id not in delete_set:
                    new_list.append((doc_id, tf))
                i += 1
            if len(new_list) > 0:
                new_postings[term] = new_list
        idx.postings = new_postings

        # Remove doc_len + doc_path + doc_meta
        i = 0
        # make a list of keys to avoid runtime size change
        keys = list(idx.doc_len.keys())
        while i < len(keys):
            did = keys[i]
            if did in delete_set:
                if did in idx.doc_len:
                    del idx.doc_len[did]
                if did in idx.doc_path:
                    del idx.doc_path[did]
                if did in idx.doc_meta:
                    del idx.doc_meta[did]
            i += 1

    # Assign new doc IDs starting from current max+1
    max_id = -1
    for did in idx.doc_path.keys():
        if did > max_id:
            max_id = did
    next_id = max_id + 1

    print("Files to (re)index:", len(to_index_paths))
    reindexed = 0
    for path in to_index_paths:
        text = read_text_file(path)
        if text is None:
            continue
        toks = tokenize(text)
        if len(toks) == 0:
            continue
        # doc id
        did = path_to_id.get(path)
        if did is None:
            did = next_id
            next_id += 1
        # update doc tables
        idx.doc_path[did] = path
        idx.doc_len[did] = len(toks)
        st = os.stat(path)
        idx.doc_meta[did] = {"mtime": float(st.st_mtime), "size": int(st.st_size)}

        # term frequencies in this doc
        tf_map: Dict[str, int] = {}
        j = 0
        while j < len(toks):
            t = toks[j]
            tf_map[t] = tf_map.get(t, 0) + 1
            j += 1

        # For every term, update postings (replace doc's tf if existed)
        for term, tf in tf_map.items():
            plist = idx.postings.get(term)
            if plist is None:
                plist = []
                idx.postings[term] = plist
            # look for existing entry
            found = False
            k = 0
            while k < len(plist):
                (pdid, _) = plist[k]
                if pdid == did:
                    plist[k] = (did, tf)
                    found = True
                    break
                k += 1
            if not found:
                plist.append((did, tf))

        reindexed += 1
        if reindexed % 50 == 0:
            print("Indexed", reindexed, "files...")

    # Recompute avg doc length
    total_len = 0
    count = 0
    for did, dlen in idx.doc_len.items():
        total_len += dlen
        count += 1
    if count > 0:
        idx.avg_doc_len = float(total_len) / float(count)
    else:
        idx.avg_doc_len = 0.0

    print("Total docs:", count, "| Avg doc len:", round(idx.avg_doc_len, 2))
    print("Saving index...")
    idx.save()
    print("Done.")

# -------------------------
# Query parsing
# -------------------------

class ParsedQuery:
    def __init__(self):
        self.must_terms: List[str] = []
        self.forbid_terms: List[str] = []
        self.optional_terms: List[str] = []
        self.phrases: List[str] = []
        self.ext_filters: Set[str] = set()

def parse_query(q: str) -> ParsedQuery:
    pq = ParsedQuery()
    q = q.strip()
    # Extract phrases "..."
    phrase_pattern = re.compile(r'"([^"]+)"')
    phrases = phrase_pattern.findall(q)
    # Remove phrases from q
    q_wo_phrases = phrase_pattern.sub(" ", q)

    i = 0
    while i < len(phrases):
        ph = phrases[i].strip()
        if len(ph) > 0:
            pq.phrases.append(normalize_text(ph))
        i += 1

    parts = q_wo_phrases.split()
    i = 0
    while i < len(parts):
        token = parts[i].strip()
        if token == "":
            i += 1
            continue
        # ext: filter
        if token.lower().startswith("ext:"):
            ext = token[4:].strip().lower()
            if not ext.startswith("."):
                ext = "." + ext
            pq.ext_filters.add(ext)
            i += 1
            continue
        # + / - terms
        if token.startswith("+") and len(token) > 1:
            t = tokenize(token[1:])
            if len(t) > 0:
                pq.must_terms.append(t[0])
            i += 1
            continue
        if token.startswith("-") and len(token) > 1:
            t = tokenize(token[1:])
            if len(t) > 0:
                pq.forbid_terms.append(t[0])
            i += 1
            continue
        # plain term -> optional
        tks = tokenize(token)
        if len(tks) > 0:
            pq.optional_terms.append(tks[0])
        i += 1

    return pq

# -------------------------
# Search (BM25 + filters)
# -------------------------

def doc_ext(path: str) -> str:
    _, ext = os.path.splitext(path)
    return ext.lower()

def idf(term: str, N: int, df: int) -> float:
    # BM25 IDF with +0.5 smoothing
    # log((N - df + 0.5) / (df + 0.5) + 1) to keep positive
    return math.log(( (N - df + 0.5) / (df + 0.5) ) + 1.0)

def bm25_score_for_doc(term_freqs: Dict[str, int], dlen: int, avgdl: float, N: int, dfs: Dict[str, int]) -> float:
    score = 0.0
    for term, tf in term_freqs.items():
        df = dfs.get(term, 0)
        if df <= 0:
            continue
        idf_val = idf(term, N, df)
        denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (float(dlen) / float(avgdl if avgdl > 0 else 1.0)))
        s = idf_val * ( (tf * (BM25_K1 + 1.0)) / (denom if denom != 0.0 else 1.0) )
        score += s
    return score

def postings_to_set(plist: List[Tuple[int, int]]) -> Set[int]:
    s: Set[int] = set()
    i = 0
    while i < len(plist):
        s.add(plist[i][0])
        i += 1
    return s

def build_term_dfs(idx: Index, terms: List[str]) -> Dict[str, int]:
    dfs: Dict[str, int] = {}
    i = 0
    while i < len(terms):
        t = terms[i]
        plist = idx.postings.get(t)
        if plist is not None:
            dfs[t] = len(plist)
        else:
            dfs[t] = 0
        i += 1
    return dfs

def phrase_in_text(phrase: str, text: str) -> bool:
    # simple substring check on normalized text
    return normalize_text(phrase) in normalize_text(text)

def passes_ext_filters(pq: ParsedQuery, path: str) -> bool:
    if len(pq.ext_filters) == 0:
        return True
    ext = doc_ext(path)
    return ext in pq.ext_filters

def collect_candidates(idx: Index, pq: ParsedQuery) -> Set[int]:
    # Start with must terms if any; otherwise union of optional; otherwise all docs
    candidates: Set[int] = set()

    if len(pq.must_terms) > 0:
        # Initialize with docs of first must term
        first = pq.must_terms[0]
        plist = idx.postings.get(first)
        if plist is None:
            return set()
        candidates = postings_to_set(plist)
        # Intersect with others
        i = 1
        while i < len(pq.must_terms):
            term = pq.must_terms[i]
            plist = idx.postings.get(term)
            if plist is None:
                return set()
            cand2 = postings_to_set(plist)
            candidates = candidates.intersection(cand2)
            if len(candidates) == 0:
                return candidates
            i += 1
    else:
        # union of optional terms
        i = 0
        while i < len(pq.optional_terms):
            term = pq.optional_terms[i]
            plist = idx.postings.get(term)
            if plist is not None:
                cand2 = postings_to_set(plist)
                if len(candidates) == 0:
                    candidates = set(cand2)
                else:
                    candidates = candidates.union(cand2)
            i += 1
        # If still empty and no terms at all, then all docs
        if len(candidates) == 0 and len(pq.optional_terms) == 0 and len(pq.phrases) == 0:
            # all docs
            for did in idx.doc_len.keys():
                candidates.add(did)

    # Remove forbidden-term docs
    i = 0
    while i < len(pq.forbid_terms):
        term = pq.forbid_terms[i]
        plist = idx.postings.get(term)
        if plist is not None:
            forb = postings_to_set(plist)
            candidates = candidates.difference(forb)
        i += 1

    # Apply ext filters
    if len(pq.ext_filters) > 0:
        filtered = set()
        for did in candidates:
            path = idx.doc_path.get(did, "")
            if passes_ext_filters(pq, path):
                filtered.add(did)
        candidates = filtered

    return candidates

def score_candidates(idx: Index, pq: ParsedQuery, candidates: Set[int]) -> List[Tuple[int, float]]:
    # Build list of all scoring terms: must + optional (phrases are filters)
    scoring_terms: List[str] = []
    i = 0
    while i < len(pq.must_terms):
        scoring_terms.append(pq.must_terms[i])
        i += 1
    i = 0
    while i < len(pq.optional_terms):
        scoring_terms.append(pq.optional_terms[i])
        i += 1

    dfs = build_term_dfs(idx, scoring_terms)
    N = len(idx.doc_len)
    if N == 0:
        return []

    results: List[Tuple[int, float]] = []

    # For each candidate, compute term_freqs for scoring terms
    for did in candidates:
        dlen = idx.doc_len.get(did, 0)
        term_freqs: Dict[str, int] = {}
        j = 0
        while j < len(scoring_terms):
            t = scoring_terms[j]
            tf = 0
            plist = idx.postings.get(t)
            if plist is not None:
                k = 0
                while k < len(plist):
                    (pdid, ptf) = plist[k]
                    if pdid == did:
                        tf = ptf
                        break
                    k += 1
            if tf > 0:
                term_freqs[t] = tf
            j += 1
        if len(term_freqs) == 0 and (len(pq.must_terms) > 0 or len(pq.optional_terms) > 0):
            continue
        score = bm25_score_for_doc(term_freqs, dlen, idx.avg_doc_len, N, dfs)
        # Light boost for more required terms satisfied
        boost = 0.0
        j = 0
        while j < len(pq.must_terms):
            t = pq.must_terms[j]
            if t in term_freqs:
                boost += 0.1
            j += 1
        score += boost
        results.append((did, score))

    # Phrase filtering (must be present)
    if len(pq.phrases) > 0:
        filtered: List[Tuple[int, float]] = []
        k = 0
        while k < len(results):
            (did, sc) = results[k]
            text = read_text_file(idx.doc_path.get(did, ""))
            ok = True
            if text is None:
                ok = False
            else:
                j = 0
                while j < len(pq.phrases):
                    if not phrase_in_text(pq.phrases[j], text):
                        ok = False
                        break
                    j += 1
            if ok:
                filtered.append((did, sc + 0.2))  # tiny bonus for phrase match
            k += 1
        results = filtered

    # Sort by score desc, tie-break by shorter path (arbitrary but stable)
    results.sort(key=lambda x: (-(x[1]), idx.doc_path.get(x[0], "")))
    return results

# -------------------------
# Snippets
# -------------------------

def build_snippet(path: str, query_terms: List[str], phrases: List[str], width: int = 200) -> str:
    text = read_text_file(path)
    if text is None or text.strip() == "":
        return ""

    norm = normalize_text(text)
    # Find earliest hit position among terms/phrases
    hits: List[int] = []

    # terms
    i = 0
    while i < len(query_terms):
        t = query_terms[i]
        # find as whole word approximate (use regex on normalized)
        try:
            pat = r"\b" + re.escape(t) + r"\b"
            m = re.search(pat, norm)
            if m:
                hits.append(m.start())
        except Exception:
            pass
        i += 1

    # phrases
    i = 0
    while i < len(phrases):
        ph = phrases[i]
        pos = norm.find(normalize_text(ph))
        if pos >= 0:
            hits.append(pos)
        i += 1

    start = 0
    if len(hits) > 0:
        first = min(hits)
        start = max(0, first - int(width / 2))

    end = min(len(text), start + width)
    snippet = text[start:end]

    # highlight naive: wrap occurrences of terms/phrases (case-insensitive)
    # Do phrases first to avoid double-wrapping inside them
    i = 0
    while i < len(phrases):
        ph = phrases[i]
        snippet = highlight(snippet, ph)
        i += 1
    i = 0
    while i < len(query_terms):
        t = query_terms[i]
        snippet = highlight(snippet, t)
        i += 1

    # add ellipses if trimmed
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet

def highlight(text: str, needle: str) -> str:
    # Case-insensitive replace with markers, preserving original casing.
    if needle.strip() == "":
        return text
    # Build regex with word boundaries if single word; else simple find
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    return pattern.sub(lambda m: HIGHLIGHT_LEFT + m.group(0) + HIGHLIGHT_RIGHT, text)

# -------------------------
# CLI / REPL
# -------------------------

def cmd_index(args):
    root = args.path
    if not os.path.isdir(root):
        print("Not a directory:", root)
        sys.exit(1)
    index_directory(root)

def cmd_search(args):
    idx = Index.load()
    if len(idx.doc_len) == 0:
        print("Index is empty. Run: python local_search.py index <folder>")
        sys.exit(1)
    pq = parse_query(args.query)
    candidates = collect_candidates(idx, pq)
    results = score_candidates(idx, pq, candidates)
    if len(results) == 0:
        print("No results.")
        return
    topn = args.top
    if topn <= 0:
        topn = 10
    shown = 0
    print("Query:", args.query)
    print("Results:", len(results))
    print("-" * 80)
    while shown < len(results) and shown < topn:
        (did, score) = results[shown]
        path = idx.doc_path.get(did, "?")
        snippet = build_snippet(path, pq.must_terms + pq.optional_terms, pq.phrases, width=args.snippet)
        print("#{0}  score={1:.3f}".format(shown + 1, score))
        print("id={0}  ext={1}  len={2}  path={3}".format(did, doc_ext(path), idx.doc_len.get(did, 0), path))
        if snippet.strip() != "":
            print("... {0}".format(snippet.replace("\n", " ")))
        print("-" * 80)
        shown += 1

def cmd_show(args):
    idx = Index.load()
    did = args.doc_id
    if did not in idx.doc_path:
        print("Unknown doc id:", did)
        return
    path = idx.doc_path[did]
    print("Doc", did, "->", path)
    text = read_text_file(path)
    if text is None:
        print("(unable to read file)")
        return
    # Print first N lines
    lines = text.splitlines()
    n = args.lines
    if n <= 0:
        n = 40
    i = 0
    while i < len(lines) and i < n:
        print(lines[i])
        i += 1
    if i < len(lines):
        print("... (truncated)")

def cmd_repl(args):
    idx = Index.load()
    if len(idx.doc_len) == 0:
        print("Index is empty. Run: python local_search.py index <folder>")
        return
    print("Local Search REPL. Type :help for commands, :quit to exit.")
    while True:
        try:
            q = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q == "":
            continue
        if q.startswith(":"):
            if q == ":quit":
                break
            if q == ":help":
                print("Commands:")
                print("  :help            show this help")
                print("  :quit            exit")
                print("  :stats           show index stats")
                print("  :files <ext>     list files with extension (.md, .txt, etc.)")
                continue
            if q.startswith(":stats"):
                total = len(idx.doc_len)
                print("Docs:", total, "| Avg doc len:", round(idx.avg_doc_len, 2), "| Terms:", len(idx.postings))
                continue
            if q.startswith(":files"):
                parts = q.split()
                if len(parts) >= 2:
                    ext = parts[1].lower()
                    if not ext.startswith("."):
                        ext = "." + ext
                    count = 0
                    for did, path in idx.doc_path.items():
                        if doc_ext(path) == ext:
                            print(did, path)
                            count += 1
                    print("Total", count, "files with", ext)
                else:
                    print("usage: :files .md")
                continue
            print("Unknown command. Type :help.")
            continue

        pq = parse_query(q)
        candidates = collect_candidates(idx, pq)
        results = score_candidates(idx, pq, candidates)
        if len(results) == 0:
            print("(no results)")
            continue
        limit = 10
        shown = 0
        while shown < len(results) and shown < limit:
            (did, score) = results[shown]
            path = idx.doc_path.get(did, "?")
            snippet = build_snippet(path, pq.must_terms + pq.optional_terms, pq.phrases, width=160)
            print("#{0}  {1:.3f}  id={2}  {3}".format(shown + 1, score, did, path))
            if snippet.strip() != "":
                print("     ", snippet.replace("\n", " "))
            shown += 1

def build_argparser():
    p = argparse.ArgumentParser(description="Local Personal Search Engine (no deps)")
    sub = p.add_subparsers()

    p_index = sub.add_parser("index", help="Index a folder (incremental)")
    p_index.add_argument("path", help="Folder path to index")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Query string")
    p_search.add_argument("--top", type=int, default=10, help="Top N results to show")
    p_search.add_argument("--snippet", type=int, default=200, help="Snippet width (chars)")
    p_search.set_defaults(func=cmd_search)

    p_show = sub.add_parser("show", help="Show a doc by id")
    p_show.add_argument("doc_id", type=int, help="Document id")
    p_show.add_argument("--lines", type=int, default=40, help="Lines to print")
    p_show.set_defaults(func=cmd_show)

    p_repl = sub.add_parser("repl", help="Interactive search mode")
    p_repl.set_defaults(func=cmd_repl)

    return p

def main():
    parser = build_argparser()
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
