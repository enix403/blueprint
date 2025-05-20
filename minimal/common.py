import torch
import torch.nn.functional as F
import networkx as nx


def conv_mask(mask, kernel, threshold_match=None):
    mask = mask.to(torch.int8).unsqueeze(0).unsqueeze(0)
    # kernel is assumed to be in int8
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    f = kernel.shape[-1]
    padding = (f - 1) // 2

    result = F.conv2d(mask, kernel, padding=padding)
    result = result[0, 0, :, :]

    if threshold_match is not None:
        result = (result == threshold_match).byte()

    return result


# -----------------------


def attr_graph_to_nx(node_attrs, edges, attr_key: str):
    G = nx.Graph()
    G.add_nodes_from([(i, {attr_key: typ}) for i, typ in enumerate(node_attrs)])
    G.add_edges_from(edges)
    return G


# -----------------------


def flatten_nx_graph(G, select_key: str, sort_key: str, reverse: bool = False):
    # Get node_labels sorted by sort_key (room types)
    # node_labels are the original index of that node in the input layout
    # e.g [4, 5, 0, 1, 2, ...]
    node_labels = sorted(G.nodes, key=lambda n: G.nodes[n][sort_key], reverse=reverse)

    # Get the node_keys (room types) corresponding to each node label above
    # e.g [LIVING, BEDROOM, ...]
    node_keys = [G.nodes[n][select_key] for n in node_labels]

    # Reindex the nodes in an increasing sequence (while still
    # maintaining corresponding order with node_labels)
    # e.g in [0, 1, 2, 3...]
    # This also appropriately renames the edges
    G = nx.relabel_nodes(G, mapping={n: i for i, n in enumerate(node_labels)}, copy=True)

    # The edges are already correctly named because of above
    # relabel_nodes(...) call
    edges = list(G.edges)

    return node_keys, edges, node_labels
