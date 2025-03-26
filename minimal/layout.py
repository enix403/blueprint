from typing import Union

from dataclasses import dataclass
import networkx as nx

# from minimal.layout import NodeType, NODE_COLOR, NODE_NAME
from minimal.common import flatten_nx_graph, attr_graph_to_nx


class NodeType:
    # Node types (rooms/doors) and their IDs from HouseGAN++
    # fmt: off
    LIVING_ROOM   = 0
    KITCHEN       = 1
    BEDROOM       = 2
    BATHROOM      = 3
    BALCONY       = 4
    ENTRANCE      = 5
    DINING_ROOM   = 6
    STUDY_ROOM    = 7
    STORAGE       = 9
    FRONT_DOOR    = 14
    INTERIOR_DOOR = 16
    # fmt: on

    # This is what the model expects
    NUM_NODE_TYPES = 18

    @classmethod
    def is_door(cls, node: int) -> bool:
        """
        Check if the given node type corresponds to a door.

        Args:
            node (int): Node type identifier.

        Returns:
            bool: True if the node represents a door, False otherwise.
        """
        return node in [cls.FRONT_DOOR, cls.INTERIOR_DOOR]

    @classmethod
    def is_room(cls, node: int) -> bool:
        """
        Check if the given node type corresponds to a room.

        Args:
            node (int): Node type identifier.

        Returns:
            bool: True if the node represents a room, False otherwise.
        """
        return not cls.is_door(node)


# fmt: off
NODE_COLOR = {
    NodeType.LIVING_ROOM   : "#EE4D4D",
    NodeType.KITCHEN       : "#C67C7B",
    NodeType.BEDROOM       : "#FFD274",
    NodeType.BATHROOM      : "#BEBEBE",
    NodeType.BALCONY       : "#BFE3E8",
    NodeType.ENTRANCE      : "#7BA779",
    NodeType.DINING_ROOM   : "#E87A90",
    NodeType.STUDY_ROOM    : "#FF8C69",
    NodeType.STORAGE       : "#1F849B",
    NodeType.FRONT_DOOR    : "#727171",
    NodeType.INTERIOR_DOOR : "#D3A2C7",
}
# fmt: on

# fmt: off
NODE_NAME = {
    NodeType.LIVING_ROOM   : "L",
    NodeType.KITCHEN       : "K",
    NodeType.BEDROOM       : "R",
    NodeType.BATHROOM      : "H",
    NodeType.BALCONY       : "A",
    NodeType.ENTRANCE      : "E",
    NodeType.DINING_ROOM   : "D",
    NodeType.STUDY_ROOM    : "S",
    NodeType.STORAGE       : "T",
    NodeType.FRONT_DOOR    : ":F",
    NodeType.INTERIOR_DOOR : ":d",
}
# fmt: on

# ------------------------


@dataclass(frozen=True)
class InputLayout:
    node_types: list[int]
    edges: list[tuple[int, int]]

    # TODO: refactor structure
    node_labels: list[int] = None

    def num_rooms(self):
        # TODO: need to optimize ?
        return len(list(filter(NodeType.is_room, self.node_types)))

    def draw(self):
        G = attr_graph_to_nx(self.node_types, self.edges, "node_type")

        nx.draw(
            G,
            nx.kamada_kawai_layout(G),
            node_size=1000,
            node_color=[NODE_COLOR[d["node_type"]] for n, d in G.nodes(data=True)],
            with_labels=True,
            labels={n: NODE_NAME[d["node_type"]] for n, d in G.nodes(data=True)},
            font_color="black",
            font_weight="bold",
            font_size=14,
            edge_color="#b9c991",
            width=2.0,
        )


def _ensure_front_door(G: nx.Graph):
    front_doors = [
        node
        for node, data in G.nodes(data=True)
        if data["node_type"] == NodeType.FRONT_DOOR
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
            node_type = data["node_type"]

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
    G.remove_nodes_from(
        [
            node
            for node, data in G.nodes(data=True)
            if (
                data["node_type"] not in NODE_NAME
                or data["node_type"] == NodeType.INTERIOR_DOOR
            )
        ]
    )

    # keep the largest component, discard rest
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    G.remove_nodes_from(comps[1:])

    if len(G) == 0:
        raise Exception("Empty input graph")

    _ensure_front_door(G)


def into_layout(node_types, edges):
    # Convert this flat structure into a nx.Graph
    G = attr_graph_to_nx(node_types, edges, "node_type")

    # Validate and clean the graph
    _clean_layout_graph(G)

    # Convert the graph back to flat lists
    node_types, edges, node_labels = flatten_nx_graph(
        G, select_key="node_type", sort_key="node_type"
    )

    layout = InputLayout(node_types, edges, node_labels)

    return layout


def into_layout_unchecked(node_types, edges):
    node_labels = list(range(len(node_types)))
    layout = InputLayout(node_types, edges, node_labels)
    return layout


# --------------------------------


class InputLayoutBuilderNode:
    type: int

    def __init__(self, type: int):
        self.type = type
        self.index = -1


class InputLayoutBuilder:
    nodes: list[InputLayoutBuilderNode]
    edges: list[tuple[InputLayoutBuilderNode, InputLayoutBuilderNode]]

    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, type):
        node = InputLayoutBuilderNode(type)
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        a: Union[InputLayoutBuilderNode, int],
        b: Union[InputLayoutBuilderNode, int],
    ):
        if isinstance(a, int):
            a = self.add_node(a)

        if isinstance(b, int):
            b = self.add_node(b)

        self.edges.append((a, b))

        return (a, b)

    def build(self) -> InputLayout:
        for i, node in enumerate(self.nodes):
            node.index = i

        return into_layout(
            list(map(lambda node: node.type, self.nodes)),
            list(map(lambda e: (e[0].index, e[1].index), self.edges)),
        )
