from typing import Union, Iterable

import copy
from dataclasses import dataclass

import torch
import networkx as nx
from matplotlib import colors

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

NODE_COLORMAP = colors.ListedColormap(
    [
        NODE_COLOR.get(i, "#FF00FF")
        for i in range(NodeType.NUM_NODE_TYPES)
    ]
)
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

class InputGraphBuilderNode:
    type: int

    def __init__(self, type: int):
        self.type = type
        self.index = -1


class InputGraphBuilder:
    nodes: list[InputGraphBuilderNode]
    edges: list[tuple[InputGraphBuilderNode, InputGraphBuilderNode]]

    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, type):
        node = InputGraphBuilderNode(type)
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        a: Union[InputGraphBuilderNode, int],
        b: Union[InputGraphBuilderNode, int],
    ):
        if isinstance(a, int):
            a = self.add_node(a)

        if isinstance(b, int):
            b = self.add_node(b)

        self.edges.append((a, b))

        return (a, b)


    def build(self) -> InputGraph:
        self.nodes.sort(key=lambda n: n.type)

        for i, node in enumerate(self.nodes):
            node.index = i

        return InputGraph(
            map(lambda node: node.type, self.nodes),
            map(lambda e: (e[0].index, e[1].index), self.edges)
        )


