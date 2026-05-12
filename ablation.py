"""
ablation.py --- single-instance, four-candidate driver for Hvala.

Loads a DIMACS-format file (or every DIMACS-format file in a directory),
runs each of the four ensemble candidates independently:

    c1 = maximal_matching_vertex_cover(G)
    c2 = bucket_degree_greedy(adj)
    c3 = covering_via_reduction_max_degree_1(G)   (Hallelujah)

prunes each one independently, and then computes the pruned union

    c4 = prune_redundant_vertices(adj, c1 | c2 | c3)

emitting a JSON record per instance to stdout. Output format:

    {"results": [
        {"name": "frb30-15-1", "n": 450, "m": 17827,
         "c1": 429, "c2": 426, "c3": 429, "c4": 430,
         "min": 426, "time_s": 0.27},
         ...
     ]}

Usage from the repository root, after `pip install hvala`:

    python ablation.py experiment/npbench/benchmarks_npbench         > frb.json
    python ablation.py experiment/npbench/clique_complement_npbench  > dimacs.json
    python ablation.py experiment/network                            > realworld.json
    python ablation.py experiment/hardest                            > hardest.json

A single file is also accepted:

    python ablation.py experiment/hardest/petersen.clq               > petersen.json

The four-candidate definition matches Algorithm 1 of the manuscript and the
reference implementation `hvala/algorithm.py`.
"""

import sys
import os
import json
import time
import glob
from collections import deque

import networkx as nx

# Re-implement the four candidates inline so that ablation.py is fully
# self-contained and does not import the (Python 3.12-pinned) hvala package.
# These mirror hvala/algorithm.py exactly.


def maximal_matching_vertex_cover(G):
    cover = set()
    M = nx.approximation.min_maximal_matching(G)
    for u, v in M:
        cover.add(u)
        cover.add(v)
    return cover


def bucket_degree_greedy(adj):
    """Linear-time max-degree greedy vertex cover via a bucket queue."""
    deg = {v: len(adj[v]) for v in adj}
    maxd = max(deg.values(), default=0)
    buckets = [deque() for _ in range(maxd + 1)]
    for v, d in deg.items():
        buckets[d].append(v)
    removed = set()
    cover = set()
    for d in reversed(range(maxd + 1)):
        q = buckets[d]
        while q:
            v = q.popleft()
            if v in removed or deg[v] != d:
                continue
            if deg[v] == 0:
                continue
            cover.add(v)
            removed.add(v)
            for u in adj[v]:
                if u not in removed:
                    deg[u] -= 1
                    buckets[deg[u]].append(u)
    return cover


def _min_weighted_vc_max_deg_1(G, weight="weight"):
    """Exact min weighted VC on a graph of max degree 1 (matching+isolated)."""
    vc = set()
    visited = set()
    for node in list(G.nodes()):
        if node in visited:
            continue
        d = G.degree(node)
        if d == 0:
            visited.add(node)
        elif d == 1:
            nb = list(G.neighbors(node))[0]
            if nb not in visited:
                nw = G.nodes[node].get(weight, 1)
                bw = G.nodes[nb].get(weight, 1)
                if nw < bw or (nw == bw and node < nb):
                    vc.add(node)
                else:
                    vc.add(nb)
                visited.add(node)
                visited.add(nb)
    return vc


def covering_via_reduction_max_degree_1(graph):
    """Hallelujah degree-1 weighted reduction; mirrors hvala/algorithm.py."""
    G = graph.copy()
    weights = {}
    for u in list(graph.nodes()):
        nbs = list(G.neighbors(u))
        G.remove_node(u)
        k = len(nbs)
        if k == 0:
            continue
        for i, v in enumerate(nbs):
            aux = (u, i)
            G.add_edge(aux, v)
            weights[aux] = 1.0 / k
    uw = _min_weighted_vc_max_deg_1(G)
    nx.set_node_attributes(G, weights, "weight")
    ww = _min_weighted_vc_max_deg_1(G)

    def back(c):
        r = set()
        for x in c:
            r.add(x[0] if isinstance(x, tuple) else x)
        return r

    us, ws = back(uw), back(ww)
    return ws if len(ws) <= len(us) else us


def prune_redundant_vertices(adj, C):
    """Linear-time pruning of redundant vertices from a vertex cover."""
    C = set(C)
    for v in list(C):
        ok = True
        for u in adj.get(v, []):
            if u not in C:
                ok = False
                break
        if ok:
            C.remove(v)
    return C


# ---------- DIMACS reader (tolerant) ----------


def load_dimacs(path):
    """Read a DIMACS edge file. Accepts `e u v` and other 3-token edge lines
    where the last two tokens are positive integers (matches the reference
    parser in hvala/parser.py)."""
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                u = int(parts[-2])
                v = int(parts[-1])
            except ValueError:
                continue
            if u <= 0 or v <= 0:
                continue
            if u == v:
                continue
            G.add_edge(u - 1, v - 1)
    return G


# ---------- per-instance ablation ----------


def ablate(path):
    """Return an ablation record for a single instance file."""
    t0 = time.time()
    G = load_dimacs(path)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m == 0:
        return {"name": os.path.basename(path), "n": n, "m": m,
                "c1": 0, "c2": 0, "c3": 0, "c4": 0, "min": 0,
                "time_s": time.time() - t0}
    adj = {v: set(G[v]) for v in G}
    c1 = maximal_matching_vertex_cover(G)
    c2 = bucket_degree_greedy(adj)
    c3 = covering_via_reduction_max_degree_1(G)
    c1 = prune_redundant_vertices(adj, c1)
    c2 = prune_redundant_vertices(adj, c2)
    c3 = prune_redundant_vertices(adj, c3)
    c4 = prune_redundant_vertices(adj, c1 | c2 | c3)
    sizes = (len(c1), len(c2), len(c3), len(c4))
    return {"name": os.path.basename(path), "n": n, "m": m,
            "c1": sizes[0], "c2": sizes[1],
            "c3": sizes[2], "c4": sizes[3],
            "min": min(sizes), "time_s": time.time() - t0}


def _expand(arg):
    """Expand `arg` into a list of file paths. Directories are recursively
    walked; non-existent .txt log files are skipped."""
    if os.path.isdir(arg):
        return sorted(
            os.path.join(arg, f)
            for f in os.listdir(arg)
            if os.path.isfile(os.path.join(arg, f)) and not f.endswith(".txt")
        )
    return sorted(glob.glob(arg))


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    files = []
    for arg in sys.argv[1:]:
        files.extend(_expand(arg))
    results = []
    for f in files:
        try:
            r = ablate(f)
            results.append(r)
        except Exception as e:
            results.append({"name": os.path.basename(f), "error": str(e)})
    json.dump({"results": results}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
