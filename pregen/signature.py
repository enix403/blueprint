import hashlib
from datasketch import MinHash, MinHashLSH

from .premade import premade_graphs

NUM_PERMS = 256

def minhash_for_graph(graph) -> MinHash:
    m = MinHash(num_perm=NUM_PERMS)

    # Encode node types
    for i, t in enumerate(graph.node_types):
        m.update(f"node-{i}-{t}".encode())

    # Canonicalize and sort edges
    canonical_edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges)

    # Encode edges as unordered
    for u, v in canonical_edges:
        m.update(f"edge-{u}-{v}".encode())

    return m


def graph_similarity(g1, g2) -> float:
    # Node type similarity (normalized positional match)
    types1, types2 = g1.node_types, g2.node_types
    match_len = min(len(types1), len(types2))
    type_score = sum(t1 == t2 for t1, t2 in zip(
        types1, types2)) / max(len(types1), len(types2))

    # Edge similarity (unordered set Jaccard)
    e1 = {frozenset([u, v]) for u, v in g1.edges}
    e2 = {frozenset([u, v]) for u, v in g2.edges}
    edge_score = len(e1 & e2) / len(e1 | e2) if e1 or e2 else 1.0

    return 0.75 * type_score + 0.25 * edge_score


def canonical_graph_signature(graph) -> str:
    # Node types (already sorted in definition)
    node_part = ','.join(map(str, graph.node_types))

    # Edges as unordered sets, then sorted
    edge_part = ','.join(
        f"{min(u, v)}-{max(u, v)}" for u, v in sorted(map(lambda e: (min(e), max(e)), graph.edges))
    )

    return f"nodes:{node_part}|edges:{edge_part}"


signature_to_graph = {}

for i, graph in enumerate(premade_graphs):
    sig = canonical_graph_signature(graph)
    signature_to_graph[sig] = graph

lsh = MinHashLSH(threshold=0.4, num_perm=NUM_PERMS)

graph_index = []  # Keep for lookup

for idx, graph in enumerate(premade_graphs):
    mh = minhash_for_graph(graph)
    lsh.insert(f"graph_{idx}", mh)
    graph_index.append(graph)


def find_closest_graph(query):
    sig = canonical_graph_signature(query)
    if sig in signature_to_graph:
        return signature_to_graph[sig]  # 🎯 Exact match

    query_mh = minhash_for_graph(query)
    candidates = lsh.query(query_mh)

    if not candidates:
        return None  # No match found

    # Rerank candidates using full similarity
    best_graph = max(
        (graph_index[int(cid.split('_')[1])] for cid in candidates),
        key=lambda g: graph_similarity(query, g)
    )
    return best_graph

def graph_folder_name(graph) -> str:
    sig = canonical_graph_signature(graph)
    full_hash = hashlib.sha256(sig.encode()).hexdigest()
    return full_hash[:20]