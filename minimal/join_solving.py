import torch
from scipy.cluster.hierarchy import DisjointSet
from minimal.rooms import RoomAreas

# --------------------

def _are_rects_adjacent(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Check if they are adjacent horizontally or vertically
    if x1 + h1 == x2 or x2 + h2 == x1:  # Vertical adjacency
        return not (y1 + w1 <= y2 or y2 + w2 <= y1)
    if y1 + w1 == y2 or y2 + w2 == y1:  # Horizontal adjacency
        return not (x1 + h1 <= x2 or x2 + h2 <= x1)
    return False

def _are_rooms_adjacent(room1, room2):
    rects1 = list(room1.iter_rects())
    rects2 = list(room2.iter_rects())

    for r1 in rects1:
        for r2 in rects2:
            if _are_rects_adjacent(r1, r2):
                return True

    return False


def spatial_adj_edges(rooms: list[RoomAreas]):
    R = len(rooms)

    edges = set()

    for i in range(R):
        for j in range(i + 1, R):
            room1 = rooms[i]
            room2 = rooms[j]

            if _are_rooms_adjacent(room1, room2):
                edges.add((j + 1, i + 1))

    return edges


def target_input_edges(
    layout_graph,
    rooms: list[RoomAreas]
):
    R = len(rooms)

    target_edges = set()

    for i in range(R):
        for j in range(i + 1, R):
            ra = j + 1
            rb = i + 1

            ra_node_index = rooms[j].room_node_index
            rb_node_index = rooms[i].room_node_index

            if layout_graph.has_edge(ra_node_index, rb_node_index):
                target_edges.add((ra, rb))

    return target_edges


# --------

def select_edges(R, available_edges, target_edges):
    # copy available edges
    available_edges = set(available_edges)
    out_edges = set()

    dsu = DisjointSet(range(1, R + 1))

    for edge in target_edges:
        if edge in available_edges:
            out_edges.add(edge)
            dsu.merge(edge[0], edge[1])
            available_edges.discard(edge)

    for edge in available_edges:
        if dsu.n_subsets == 1:
            break
            
        a, b = edge
        if dsu[a] != dsu[b]:
            dsu.merge(a, b)
            out_edges.add(edge)

    return out_edges

def select_rooms_to_join(rooms, input_graph):
    R = len(rooms)
    available_edges = spatial_adj_edges(rooms)
    target_edges = target_input_edges(input_graph, rooms)

    return list(select_edges(
        R,
        available_edges, target_edges
    ))
