from dataclasses import dataclass
import networkx as nx

from minimal.layout import NodeType, NODE_COLOR, NODE_NAME
from minimal.common import flatten_nx_graph, attr_graph_to_nx

@dataclass(frozen=True)
class InputLayout:
    node_types: list[int]
    edges: list[tuple[int, int]]

    def num_rooms(self):
        # TODO: need to optimize ?
        return len(list(filter(NodeType.is_room, self.node_types)))

    def draw(self):
        G = attr_graph_to_nx(
            self.node_types,
            self.edges,
            'node_type'
        )

        nx.draw(
            G,
            nx.kamada_kawai_layout(G),
            node_size=1000,
            node_color=[
                NODE_COLOR[d['node_type']]
                for n, d in G.nodes(data=True)
            ],
            with_labels=True,
            labels={
                n: NODE_NAME[d['node_type']]
                for n, d in G.nodes(data=True)
            },
            font_color="black",
            font_weight="bold",
            font_size=14,
            edge_color="#b9c991",
            width=2.0,
        )


def _ensure_front_door(G: nx.Graph):
    front_doors = [
        node for
        node, data in G.nodes(data=True)
        if data['node_type'] == NodeType.FRONT_DOOR
    ]

    if len(front_doors) > 1:
        # If more than one front doors are present, then
        # delete the extra ones
        G.remove_nodes_from(front_doors[1:])

    elif len(front_doors) == 0:
        # else if none are present, then add one connected
        # to a room (assuming at least one room is present)

        # We connect the front door to the most "important"
        # room (e.g Living room) among the available nodes.
        # The node IDs are assigned such that the most
        # "important" room gets the lowest id.

        min_node = -1
        min_node_type = -1
        next_node = -1

        for node, data in G.nodes(data=True):
            next_node = max(node, next_node)
            node_type = data['node_type']

            if not NodeType.is_room(node_type):
                continue

            if min_node == -1 or node_type < min_node_type:
                min_node = node
                min_node_type = node_type

        next_node += 1
        G.add_node(next_node, node_type=NodeType.FRONT_DOOR)
        G.add_edge(next_node, min_node)


def _clean_layout_graph(G: nx.Graph):
    # remove invalid nodes
    G.remove_nodes_from([
        node
        for node, data in G.nodes(data=True)
        if (
            data['node_type'] not in NODE_NAME
            or data['node_type'] == NodeType.INTERIOR_DOOR
        )
    ])

    # keep the largest component, discard rest
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    G.remove_nodes_from(comps[1:])

    if len(G) == 0:
        raise Exception("Empty input graph")

    _ensure_front_door(G)


def into_layout(node_types, edges):
    # Convert this flat structure into a nx.Graph
    G = attr_graph_to_nx(node_types, edges, 'node_type')

    # Validate and clean the graph
    _clean_layout_graph(G)

    # Convert the graph back to flat lists
    node_types, edges = flatten_nx_graph(
        G,
        select_key='node_type',
        sort_key='node_type'
    )

    return InputLayout(node_types, edges)


def into_layout_unchecked(node_types, edges):
    return InputLayout(node_types, edges)