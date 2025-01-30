import torch
import networkx as nx

from minimal.layout import NodeType
from minimal.gen import PlanMasks

def _largest_rectangle_area(heights):
    """
    Find the largest rectangle area in a histogram.
    
    Args:
        heights (list[int]): A list of heights of histogram bars.

    Returns:
        int: The largest rectangle area.
        int: The start index of the rectangle.
        int: The width of the rectangle.
    """
    stack = []  # Stack to store indices of histogram bars
    max_area = 0
    start_idx = 0
    max_width = 0

    heights.append(0)  # Append a sentinel value to flush the stack
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            if area > max_area:
                max_area = area
                start_idx = stack[-1] + 1 if stack else 0
                max_width = width
        stack.append(i)

    heights.pop()  # Remove the sentinel value
    return max_area, start_idx, max_width


def _find_largest_rectangle(grid):
    """
    Find the largest rectangle of 1's in a 2D binary grid using the histogram technique.

    Args:
        grid (torch.Tensor): A 2D binary grid (n x m) of dtype torch.uint8.

    Returns:
        tuple: (x, y, width, height) where x, y are the top-left coordinates,
               width is the width of the rectangle, and height is the height.
    """
    n, m = grid.shape
    heights = torch.zeros(m, dtype=torch.int32)
    max_rectangle = (0, 0, 0, 0)
    max_area = 0

    for i in range(n):
        for j in range(m):
            heights[j] = heights[j] + 1 if grid[i, j] == 1 else 0

        area, start_col, width = _largest_rectangle_area(heights.tolist())
        if area > max_area:
            max_area = area
            max_rectangle = (i - area // width + 1, start_col, width, area // width)

    return max_rectangle


def _split_into_rectangles(grid):
    """
    Split a binary grid into the smallest set of non-overlapping rectangles.

    Args:
        grid (torch.Tensor): A 2D binary grid (n x m) of dtype torch.uint8.

    Returns:
        list[tuple]: A list of rectangles represented as (x, y, width, height).
    """
    grid = grid.clone()  # Work on a copy to avoid modifying the input
    rectangles = []

    while grid.any():
        x, y, width, height = _find_largest_rectangle(grid)
        rectangles.append((x, y, width, height))

        # Set the cells of the found rectangle to 0
        grid[x:x + height, y:y + width] = 0

    return rectangles

# ---------

def _create_rects_graph(rectangles):
    """
    Create a graph where each rectangle is a node, and an edge exists between two nodes if their rectangles touch at boundary.

    Args:
        rectangles (list[tuple]): List of rectangles represented as (x, y, width, height).

    Returns:
        networkx.Graph: A graph where nodes are rectangle indices and edges represent adjacency.
    """
    def is_adjacent(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Check if they are adjacent horizontally or vertically
        if x1 + h1 == x2 or x2 + h2 == x1:  # Vertical adjacency
            return not (y1 + w1 <= y2 or y2 + w2 <= y1)
        if y1 + w1 == y2 or y2 + w2 == y1:  # Horizontal adjacency
            return not (x1 + h1 <= x2 or x2 + h2 <= x1)
        return False

    G = nx.Graph()
    for i, rect1 in enumerate(rectangles):
        G.add_node(i, xywh=rect1)
        for j, rect2 in enumerate(rectangles):
            if i != j and is_adjacent(rect1, rect2):
                G.add_edge(i, j)

    return G


class RoomAreas:
    """
    A class to represent the area of rooms as a graph of axis aligned rectangles.
    """

    # Height of the grid
    grid_height: int

    # Width of the grid
    grid_width: int

    # Type of this room node
    room_type: int

    # Graph of rectangles
    rects_graph: nx.Graph

    def __init__(self, room_type: int, mask: torch.tensor):
        """
        Build a RoomAreas instance.

        Args:
            room_type (int): Type of this room node
            mask (torch.Tensor): A binary grid (n x m) where 1 represents the room area.
        """
        self.grid_height = mask.shape[0]
        self.grid_width = mask.shape[1]

        self.room_type = room_type

        rects = _split_into_rectangles(mask)
        self.rects_graph = _create_rects_graph(rects)

    def to_mask(self):
        """
        Convert the rectangles stored in the graph back into a binary mask.

        Returns:
            torch.Tensor: A binary grid (n x m) reconstructed from the rectangles.
        """
        mask = torch.zeros(
            (self.grid_height, self.grid_width),
            dtype=torch.uint8
        )

        for i, data in self.rects_graph.nodes(data=True):
            x, y, w, h = data['xywh']
            mask[x:x + h, y:y + w] = 1

        return mask


    # def scale_by(self, scale_height: int, scale_width: int):
    #     # TODO: make sure everything is int
    #     self.grid_height *= scale_height
    #     self.grid_width *= scale_width

    #     G = self.rects_graph

    #     for node in G.nodes:
    #         x, y, w, h = G.nodes[node]['xywh']
            
    #         x *= scale_height
    #         h *= scale_height

    #         y *= scale_width
    #         w *= scale_width

    #         G.nodes[node]['xywh'] = (x, y, w, h)


    def _area_of(self, node: int):
        _, _, w, h = self.rects_graph.nodes[node]['xywh']
        return w * h


    def total_area(self):
        return sum(map(self._area_of, self.rects_graph.nodes))
        

    def threshold_rectangles(self, min_area_units: int):
        nodes_to_remove = []

        G = self.rects_graph

        for node in G.nodes:
            _, _, w, h = G.nodes[node]['xywh']

            if w * h < min_area_units:
                nodes_to_remove.append(node)

        G.remove_nodes_from(nodes_to_remove)


    def discard_small_components(self):
        # find the largest connected components 

        comps = list(nx.connected_components(self.rects_graph))

        max_comp_i = -1
        max_area = 0

        for i, nodes in enumerate(comps):
            total_area = sum(map(self._area_of, nodes))

            if max_comp_i == -1 or total_area > max_area:
                max_area = total_area
                max_comp_i = i

        # remove others nodes from smaller components

        nodes_to_remove = []

        for i, nodes in enumerate(comps):
            if i != max_comp_i:
                nodes_to_remove.extend(nodes)

        self.rects_graph.remove_nodes_from(nodes_to_remove)


    def iter_rects(self):
        for node in self.rects_graph.nodes:
            yield self.rects_graph.nodes[node]['xywh']


def extract_rooms(pm: PlanMasks):
    # TODO: Handle empty rooms/rects throughout

    # TODO: take input
    min_area_units = 10

    rooms = []
    for i, node in enumerate(pm.graph.nodes):
        if not NodeType.is_room(node):
            continue

        room = RoomAreas(node, pm.masks[i])

        # remove short rectangles
        # TODO: maybe keep rectangles with degree >= 2 as they
        #       will prove to be a "pathway" between multiple
        #       (potentially) disconnected rooms 
        room.threshold_rectangles(min_area_units)

        # Only keep the largest connected component (by
        # commulative area)
        room.discard_small_components()

        rooms.append(room)

    return rooms
