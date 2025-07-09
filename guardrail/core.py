# guardrail/core.py — CLI **and** programmatic API
# ====================================================
"""Guardrail: zero‑false‑positive drift detection via DAFSA + BK‑trees.

Python interface
----------------
>>> import pandas as pd, guardrail as gr
>>> df = pd.read_parquet("quarter_1.parquet")
>>> gr.snapshot(df, "baseline", cols=["hospital_name"])
>>> anomalies = gr.check(pd.read_parquet("quarter_2.parquet"), "baseline", cols=["hospital_name"])
>>> anomalies["hospital_name"].brand_new[:5]

CLI
---
    guardrail snapshot --input quarter_1.parquet --cols hospital_name --baseline baseline/
    guardrail check    --input quarter_2.parquet --cols hospital_name --baseline baseline/
"""

from __future__ import annotations

import argparse, json, os, pickle, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List

import pandas as pd

###############################################################################
# ✨  Data structures: DAFSA + BK‑tree
###############################################################################
class DAFSANode:
    __slots__ = ("final", "edges")

    def __init__(self, final: bool = False):
        self.final = final
        self.edges: dict[str, "DAFSANode"] = {}


def _build_trie(words: Iterable[str]) -> DAFSANode:
    root = DAFSANode()
    for w in words:
        node = root
        for ch in w:
            node = node.edges.setdefault(ch, DAFSANode())
        node.final = True
    return root


def _minimize(node: DAFSANode, registry: dict) -> DAFSANode:
    for ch, child in list(node.edges.items()):
        node.edges[ch] = _minimize(child, registry)
    sig = (node.final, tuple(sorted((ch, id(c)) for ch, c in node.edges.items())))
    if sig in registry:
        return registry[sig]
    registry[sig] = node
    return node


def _serialize_dafsa(root: DAFSANode) -> list[dict]:
    index: dict[DAFSANode, int] = {}
    nodes: list[dict] = []

    def dfs(n: DAFSANode):
        if n in index:
            return
        idx = len(nodes)
        index[n] = idx
        nodes.append(None)  # placeholder
        for child in n.edges.values():
            dfs(child)

    dfs(root)
    for n, idx in index.items():
        nodes[idx] = {
            "final": n.final,
            "edges": {ch: index[c] for ch, c in n.edges.items()},
        }
    return nodes


def _dafsa_contains(nodes: list[dict], word: str) -> bool:
    idx = 0
    for ch in word:
        idx = nodes[idx]["edges"].get(ch, -1)
        if idx == -1:
            return False
    return nodes[idx]["final"]


# ---------------------------- BK‑tree ----------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins, dele, sub = prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)
            curr.append(min(ins, dele, sub))
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

def _snapshot_categorical(series: pd.Series, out_dir: Path, col: str):
    tokens = series.dropna().astype(str).unique().tolist()
    trie = _build_trie(tokens)
    _minimize(trie, {})
    nodes = _serialize_dafsa(trie)
    bk = BKTree(_levenshtein, tokens)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(nodes, open(out_dir / f"{col}.dafsa.json", "w"))
    pickle.dump(bk, open(out_dir / f"{col}.bk", "wb"))


def _check_categorical(series: pd.Series, base_dir: Path, col: str, max_dist: int):
    nodes = json.load(open(base_dir / f"{col}.dafsa.json"))
    bk: BKTree = pickle.load(open(base_dir / f"{col}.bk", "rb"))
    renames, brand_new = [], []
    for tok in series.dropna().astype(str):
        if _dafsa_contains(nodes, tok):
            continue
        near = bk.query(tok, max_dist)
        if near:
            renames.append((tok, near[:3]))
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


def snapshot(df: pd.DataFrame, baseline: str | Path, *, cols: list[str]):
    """Create baseline artifacts for the specified *categorical* columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the baseline batch.
    baseline : str or pathlib.Path
        Directory where snapshot artifacts will be written.
    cols : list[str]
        Column names to snapshot (categorical strings).
    """
    out_dir = Path(baseline)
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from DataFrame")
        _snapshot_categorical(df[col], out_dir, col)


def check(df: pd.DataFrame, baseline: str | Path, *, cols: list[str], distance: int = 2) -> Dict[str, DriftResult]:
    """Validate *df* against an existing baseline.

    Returns
    -------
    dict mapping column name → DriftResult
    """
    base_dir = Path(baseline)
    results: Dict[str, DriftResult] = {}
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from DataFrame")
        renames, new = _check_categorical(df[col], base_dir, col, distance)
        results[col] = DriftResult(renames, new)
    return results

__all__ = ["snapshot", "check", "DriftResult"]

###############################################################################
# 🚦  CLI – thin wrapper around the Python API
###############################################################################

def _parse_args():
    p = argparse.ArgumentParser("guardrail: zero‑FP drift guardrails")
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
