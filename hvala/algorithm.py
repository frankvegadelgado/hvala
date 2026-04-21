# Created on 26/07/2025
# Author: Frank Vega

import itertools
from . import utils

import networkx as nx
from collections import deque


# ============================================================
# 1. Maximal matching (2-approx)
# ============================================================

def maximal_matching_vertex_cover(G):
    cover = set()
    min_maximal_matching = nx.approximation.min_maximal_matching(G)
    for u, v in min_maximal_matching:
        cover.add(u)
        cover.add(v)
    return cover

# ============================================================
# 2 & 3. Bucket-queue max-degree greedy (linear-time 2-approx)
# ============================================================

def bucket_degree_greedy(adj):
    """
    Linear-time max-degree greedy vertex cover.
    """
    deg = {v: len(adj[v]) for v in adj}
    maxd = max(deg.values(), default=0)
    buckets = [deque() for _ in range(maxd + 1)]
    for v, d in deg.items():
        buckets[d].append(v)

    removed = set()
    cover = set()

    # Process from highest to lowest degree (guarantees validity)
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


# ============================================================
# 4. Weighted reduction to (near) degree-1 instance
# ============================================================

def min_weighted_vertex_cover_max_degree_1(G, weight='weight'):
    """
    Solver used by the reduction (works on the star-like auxiliary graph).
    """
    vertex_cover = set()
    visited = set()
    for node in list(G.nodes()):
        if node in visited:
            continue
        degree = G.degree(node)
        if degree == 0:
            visited.add(node)
        elif degree == 1:
            neighbor = list(G.neighbors(node))[0]
            if neighbor not in visited:
                node_weight = G.nodes[node].get(weight, 1)
                neighbor_weight = G.nodes[neighbor].get(weight, 1)
                if (node_weight < neighbor_weight or
                    (node_weight == neighbor_weight and node < neighbor)):
                    vertex_cover.add(node)
                else:
                    vertex_cover.add(neighbor)
                visited.add(node)
                visited.add(neighbor)
    return vertex_cover


def covering_via_reduction_max_degree_1(graph):
    """
    Linear-time reduction heuristic.
    Creates one auxiliary vertex per original edge and solves the resulting
    (star-shaped) instance twice (unweighted + weighted 1/d). Maps back to
    original vertices.
    """
    G = graph.copy()
    weights = {}

    for u in list(graph.nodes()):
        neighbors = list(G.neighbors(u))
        G.remove_node(u)
        k = len(neighbors)
        if k == 0:
            continue
        for i, v in enumerate(neighbors):
            aux_vertex = (u, i)
            G.add_edge(aux_vertex, v)
            weights[aux_vertex] = 1.0 / k

    # Unweighted solve
    unweighted_cover = min_weighted_vertex_cover_max_degree_1(G)

    # Weighted solve
    nx.set_node_attributes(G, weights, 'weight')
    weighted_cover = min_weighted_vertex_cover_max_degree_1(G)

    # Map back to original vertices
    def map_back(cover):
        res = set()
        for x in cover:
            if isinstance(x, tuple):
                res.add(x[0])
            else:
                res.add(x)
        return res

    unweighted_sol = map_back(unweighted_cover)
    weighted_sol = map_back(weighted_cover)

    return weighted_sol if len(weighted_sol) <= len(unweighted_sol) else unweighted_sol


# ============================================================
# Linear-time redundant-vertex pruning (replaces bitsets + local search)
# ============================================================

def prune_redundant_vertices(adj, C):
    """
    Linear-time single-pass removal of redundant vertices.
    For every v in C we check (in O(deg(v)) time) whether all its neighbors
    are still in the current cover. If yes, we safely remove it immediately.
    Total time across all calls remains O(n + m).
    """
    C = set(C)
    for v in list(C):          # list() protects against modification during iteration
        # Check if every neighbor is still in C
        all_neighbors_covered = True
        for u in adj.get(v, []):
            if u not in C:
                all_neighbors_covered = False
                break
        if all_neighbors_covered:
            C.remove(v)
    return C


# ============================================================
# Main ensemble (now strictly linear-time O(n + m))
# ============================================================

def find_vertex_cover(graph):
    G = graph.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_edges() == 0:
        return set()
    
    adj = {v: set(G[v]) for v in G}

    # 1–3: all linear-time heuristics producing valid covers
    c1 = maximal_matching_vertex_cover(G) # min-maximal-matching
    c2 = bucket_degree_greedy(adj)  # max-degree
    c3 = covering_via_reduction_max_degree_1(G) # reduction-based solver

    # 4: prune on the union (strongest starting cover)
    c4 = prune_redundant_vertices(adj, c1 | c2 | c3)
    
    # Final pruning on every candidate (still linear)
    c1 = prune_redundant_vertices(adj, c1)
    c2 = prune_redundant_vertices(adj, c2)
    c3 = prune_redundant_vertices(adj, c3)
    c4 = prune_redundant_vertices(adj, c4)

    return min([c1, c2, c3, c4], key=len)


def find_vertex_cover_brute_force(graph):
    """
    Computes an exact minimum vertex cover in exponential time.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the exact vertex cover, or None if the graph is empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    working_graph = graph.copy()
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))
    working_graph.remove_nodes_from(list(nx.isolates(working_graph)))
    
    if working_graph.number_of_nodes() == 0:
        return set()

    n_vertices = len(working_graph.nodes())

    for k in range(1, n_vertices + 1): # Iterate through all possible sizes of the cover
        for candidate in itertools.combinations(working_graph.nodes(), k):
            cover_candidate = set(candidate)
            if utils.is_vertex_cover(working_graph, cover_candidate):
                return cover_candidate
                
    return None



def find_vertex_cover_approximation(graph):
    """
    Computes an approximate vertex cover in polynomial time with an approximation ratio of at most 2 for undirected graphs.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the approximate vertex cover, or None if the graph is empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    #networkx doesn't have a guaranteed minimum vertex cover function, so we use approximation
    vertex_cover = nx.approximation.vertex_cover.min_weighted_vertex_cover(graph)
    return vertex_cover