"""driftfx: zero‑false‑positive drift detection via set + BK‑trees.

Python interface
----------------
>>> import pandas as pd, driftfx as dr
>>> df = pd.read_parquet("quarter_1.parquet")
>>> dr.snapshot(df, "baseline", cols=["hospital_name"])
>>> anomalies = dr.check(pd.read_parquet("quarter_2.parquet"), "baseline", cols=["hospital_name"])
>>> anomalies["hospital_name"].brand_new[:5]

CLI
---
    driftfx snapshot --input quarter_1.parquet --cols hospital_name --baseline baseline/
    driftfx check    --input quarter_2.parquet --cols hospital_name --baseline baseline/
"""

from __future__ import annotations

import argparse, json, os, pickle, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List
from multiprocessing import Pool, cpu_count
from functools import partial
import random

import pandas as pd

###############################################################################
# ✨  Data structures: Simple set + BK‑tree  
###############################################################################


# ---------------------------- BK‑tree ----------------------------------------

# Try to import Cython-optimized version, fall back to pure Python
try:
    from driftfx._cython.levenshtein import levenshtein as _levenshtein_cython
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False

if _USE_CYTHON:
    _levenshtein = _levenshtein_cython
else:
    def _levenshtein(s1: str, s2: str) -> int:
        # Quick returns
        if s1 == s2:
            return 0
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)
        
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        # Use simple list - faster than numpy for small strings
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                if c1 == c2:
                    curr.append(prev[j])
                else:
                    curr.append(min(prev[j + 1] + 1,  # deletion
                                  curr[j] + 1,       # insertion
                                  prev[j] + 1))      # substitution
            prev = curr
        return prev[-1]


class BKTree:
    def __init__(self, distfn, words: Iterable[str] | None = None):
        self.distfn = distfn
        self.tree: list | None = None
        if words:
            for w in words:
                self.add(w)

    def add(self, w: str):
        if self.tree is None:
            self.tree = [w, {}]
            return
        node = self.tree
        while True:
            parent, children = node
            d = self.distfn(w, parent)
            if d in children:
                node = children[d]
            else:
                children[d] = [w, {}]
                break

    def query(self, w: str, max_dist: int = 2) -> list[str]:
        if self.tree is None:
            return []
        res, stack = [], [self.tree]
        while stack:
            parent, children = stack.pop()
            d = self.distfn(w, parent)
            if d <= max_dist:
                res.append(parent)
            lo, hi = d - max_dist, d + max_dist
            for dist, child in children.items():
                if lo <= dist <= hi:
                    stack.append(child)
        return res

###############################################################################
# 🚀  Snapshot & Check helpers – internal, but reusable
###############################################################################



def _build_bktree_worker(tokens_chunk: list[str]) -> BKTree:
    """Worker function to build a BK-tree from a chunk of tokens."""
    # Shuffle for better tree balance
    shuffled = tokens_chunk.copy()
    random.shuffle(shuffled)
    return BKTree(_levenshtein, shuffled)


def _snapshot_categorical(series: pd.Series, out_dir: Path, col: str):
    # Get unique tokens
    tokens = series.dropna().astype(str).unique().tolist()
    
    # No need to build complex data structures for exact matching
    # We'll just save the tokens as a list and load as a set
    
    # Decide whether to use parallel BK-tree construction
    if len(tokens) < 1000:
        # Small dataset: single BK-tree
        shuffled_tokens = tokens.copy()
        random.shuffle(shuffled_tokens)
        bk_forest = [BKTree(_levenshtein, shuffled_tokens)]
    else:
        # Large dataset: parallel BK-tree forest
        n_trees = min(cpu_count(), 8, len(tokens) // 100)  # At least 100 tokens per tree
        chunk_size = len(tokens) // n_trees
        chunks = []
        
        for i in range(n_trees):
            start = i * chunk_size
            end = start + chunk_size if i < n_trees - 1 else len(tokens)
            chunks.append(tokens[start:end])
        
        # Build BK-trees in parallel
        with Pool(n_trees) as pool:
            bk_forest = pool.map(_build_bktree_worker, chunks)
    
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write files with context managers
    # Save tokens as simple list (will be loaded as set for O(1) lookups)
    with open(out_dir / f"{col}.baseline.json", "w") as f:
        json.dump(tokens, f)
    with open(out_dir / f"{col}.bk", "wb") as f:
        pickle.dump(bk_forest, f)


def _query_bktree_forest(bk_forest: list[BKTree], token: str, max_dist: int) -> list[str]:
    """Query multiple BK-trees and merge results."""
    all_matches = []
    for bk in bk_forest:
        matches = bk.query(token, max_dist)
        all_matches.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for match in all_matches:
        if match not in seen:
            seen.add(match)
            unique_matches.append(match)
    
    return unique_matches


def _check_token_worker(args: tuple) -> tuple[str, list[str] | None]:
    """Worker function to check a single token."""
    tok, baseline_set, bk_forest, max_dist = args
    if tok in baseline_set:
        return tok, None
    
    near = _query_bktree_forest(bk_forest, tok, max_dist)
    return tok, near[:3] if near else []


def _check_categorical(series: pd.Series, base_dir: Path, col: str, max_dist: int):
    # Load baseline data with context managers
    with open(base_dir / f"{col}.baseline.json") as f:
        baseline_set = set(json.load(f))
    with open(base_dir / f"{col}.bk", "rb") as f:
        bk_forest = pickle.load(f)
    
    renames, brand_new = [], []
    
    # Get unique values to check (avoid checking duplicates)
    unique_tokens = series.dropna().astype(str).unique()
    
    # Filter out tokens that exist in baseline first (fast path)
    anomalies = [tok for tok in unique_tokens if tok not in baseline_set]
    
    if not anomalies:
        return renames, brand_new
    
    # Check anomalies for potential renames
    if len(anomalies) < 50:
        # Small number of anomalies: check sequentially
        for tok in anomalies:
            near = _query_bktree_forest(bk_forest, tok, max_dist)
            if near:
                renames.append((tok, near[:3]))
            else:
                brand_new.append(tok)
    else:
        # Large number of anomalies: check in parallel
        with Pool(min(cpu_count(), 4)) as pool:
            tasks = [(tok, baseline_set, bk_forest, max_dist) for tok in anomalies]
            results = pool.map(_check_token_worker, tasks)
        
        for tok, near in results:
            if near is None:
                continue  # Was in baseline
            elif near:
                renames.append((tok, near))
            else:
                brand_new.append(tok)
    
    return renames, brand_new

###############################################################################
# 🛠️  Public Python API
###############################################################################
@dataclass
class DriftResult:
    renames: list[Tuple[str, list[str]]]
    brand_new: list[str]

    def is_clean(self) -> bool:
        return not self.renames and not self.brand_new


def _snapshot_worker(args: tuple) -> None:
    """Worker function for parallel snapshot processing."""
    series, out_dir, col = args
    _snapshot_categorical(series, out_dir, col)


def snapshot(df: pd.DataFrame, baseline: str | Path, *, cols: list[str], parallel: bool = True):
    """Create baseline artifacts for the specified *categorical* columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the baseline batch.
    baseline : str or pathlib.Path
        Directory where snapshot artifacts will be written.
    cols : list[str]
        Column names to snapshot (categorical strings).
    parallel : bool
        Whether to process columns in parallel (default: True).
    """
    out_dir = Path(baseline)
    
    # Validate all columns exist
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from DataFrame")
    
    if parallel and len(cols) > 1:
        # Process columns in parallel
        with Pool(min(cpu_count(), len(cols))) as pool:
            tasks = [(df[col], out_dir, col) for col in cols]
            pool.map(_snapshot_worker, tasks)
    else:
        # Sequential processing
        for col in cols:
            _snapshot_categorical(df[col], out_dir, col)


def _check_worker(args: tuple) -> tuple[str, DriftResult]:
    """Worker function for parallel check processing."""
    series, base_dir, col, distance = args
    renames, new = _check_categorical(series, base_dir, col, distance)
    return col, DriftResult(renames, new)


def check(df: pd.DataFrame, baseline: str | Path, *, cols: list[str], distance: int = 2, parallel: bool = True) -> Dict[str, DriftResult]:
    """Validate *df* against an existing baseline.

    Returns
    -------
    dict mapping column name → DriftResult
    """
    base_dir = Path(baseline)
    
    # Validate all columns exist
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from DataFrame")
    
    if parallel and len(cols) > 1:
        # Process columns in parallel
        with Pool(min(cpu_count(), len(cols))) as pool:
            tasks = [(df[col], base_dir, col, distance) for col in cols]
            results_list = pool.map(_check_worker, tasks)
        results = dict(results_list)
    else:
        # Sequential processing
        results: Dict[str, DriftResult] = {}
        for col in cols:
            renames, new = _check_categorical(df[col], base_dir, col, distance)
            results[col] = DriftResult(renames, new)
    
    return results

__all__ = ["snapshot", "check", "DriftResult"]

###############################################################################
# 🚦  CLI – thin wrapper around the Python API
###############################################################################

def _parse_args():
    p = argparse.ArgumentParser("driftfx: zero‑FP drift guardrails")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="Create baseline snapshot")
    s.add_argument("--input", required=True)
    s.add_argument("--cols", nargs="+", required=True)
    s.add_argument("--baseline", required=True)

    c = sub.add_parser("check", help="Check new data for drift")
    c.add_argument("--input", required=True)
    c.add_argument("--cols", nargs="+", required=True)
    c.add_argument("--baseline", required=True)
    c.add_argument("--distance", type=int, default=2)
    return p.parse_args()


def main():
    args = _parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit("Input file not found")

    # Lazy load
    df = pd.read_parquet(in_path) if in_path.suffix != ".csv" else pd.read_csv(in_path)

    if args.cmd == "snapshot":
        snapshot(df, args.baseline, cols=args.cols)
        print("Snapshot complete ✓")
    else:  # check
        res = check(df, args.baseline, cols=args.cols, distance=args.distance)
        total_anomalies = sum(len(r.renames) + len(r.brand_new) for r in res.values())
        for col, r in res.items():
            if r.is_clean():
                print(f"[✔] {col}: no drift")
            else:
                print(f"[✖] {col}: {len(r.renames)} renames / {len(r.brand_new)} new codes")
        if total_anomalies:
            sys.exit(f"Drift detected: {total_anomalies} anomalies")
        print("All clear ✓")

if __name__ == "__main__":
    main()
