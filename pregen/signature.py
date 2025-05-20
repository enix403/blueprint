import hashlib
import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance

from .premade import premade_graphs


def to_nx_graph(graph):
    G = nx.Graph()
    for i, t in enumerate(graph.node_types):
        G.add_node(i, label=t)
    for u, v in graph.edges:
        G.add_edge(u, v)
    return G


def canonical_graph_signature(graph):
    G = to_nx_graph(graph)
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='label')


signature_to_graph = {}

for graph in premade_graphs:
    sig = canonical_graph_signature(graph)
    signature_to_graph[sig] = graph


# For fallback similarity using GED
def graph_similarity(query, other):
    G1 = to_nx_graph(query)
    G2 = to_nx_graph(other)
    dist = graph_edit_distance(
        G1, G2,
        node_subst_cost=lambda a, b: 0 if a['label'] == b['label'] else 1,
        node_del_cost=lambda _: 1,
        node_ins_cost=lambda _: 1,
        edge_del_cost=lambda _: 1,
        edge_ins_cost=lambda _: 1,
    )
    if dist is None:
        return 0  # fallback if GED couldn't be computed
    return 1 / (1 + dist)  # Convert distance to similarity


def find_closest_graph(query, min_similarity=0.25):
    sig = canonical_graph_signature(query)
    if sig in signature_to_graph:
        return signature_to_graph[sig]  # 🎯 Exact match

    num_query_nodes = len(query.node_types)
    # Otherwise, search all graphs by GED (expensive)
    best_graph = None
    best_score = -1
    for g in premade_graphs:
        if len(g.node_types) != num_query_nodes:
            continue
        sim = graph_similarity(query, g)
        if sim > best_score:
            best_score = sim
            best_graph = g

    if best_score < min_similarity:
        return None

    return best_graph


def graph_folder_name(graph) -> str:
    # Node types (already sorted in definition)
    node_part = ','.join(map(str, graph.node_types))

    # Edges as unordered sets, then sorted
    edge_part = ','.join(
        f"{min(u, v)}-{max(u, v)}" for u, v in sorted(map(lambda e: (min(e), max(e)), graph.edges))
    )

    sig = f"nodes:{node_part}|edges:{edge_part}"

    full_hash = hashlib.sha256(sig.encode()).hexdigest()
    return full_hash[:20]