import torch
from scipy.cluster.hierarchy import DisjointSet

from minimal.rooms import RoomAreas
from minimal.walls import (
    CC_TL, CC_TR, CC_BR, CC_BL,
    CC_T,  CC_R,  CC_B,  CC_L
)

from minimal.walls import _conv_mask

def extract_face_walls(sep_mask):
    sp = sep_mask

    up_walls = (
           (sp & CC_T).bool() 
        & ~(sp & CC_L).bool()
        & ~(sp & CC_R).bool()
        & ~(sp & CC_B).bool()
    )

    right_walls = (
           (sp & CC_R).bool() 
        & ~(sp & CC_L).bool()
        & ~(sp & CC_T).bool()
        & ~(sp & CC_B).bool()
    )

    down_walls = (
           (sp & CC_B).bool() 
        & ~(sp & CC_L).bool()
        & ~(sp & CC_T).bool()
        & ~(sp & CC_R).bool()
    )

    left_walls = (
           (sp & CC_L).bool() 
        & ~(sp & CC_B).bool()
        & ~(sp & CC_T).bool()
        & ~(sp & CC_R).bool()
    )

    return [up_walls, right_walls, down_walls, left_walls]

# --------------------

_res_kernel = torch.tensor([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
], dtype=torch.int8)

# Keep walls that connect room ra to rb only and none else
def _restrict_touching(
    inv_room_mask,
    room_mask,
    ra_walls,
    ra, rb
):
    room_mask = room_mask + 1
    room_mask *= inv_room_mask[ra - 1]
    room_mask *= inv_room_mask[rb - 1]

    restricted = (_conv_mask(room_mask, _res_kernel) == 0).byte()
    restricted *= ra_walls

    return restricted

# --------------------

def _extract_walls_runs(lx, ly, min_len: int=4):
    
    lx, idx = torch.sort(lx)
    ly = ly[idx]
    
    runs = []

    prev_x = lx[0]
    prev_y = ly[0]
    cur_len = 1

    for x, y in zip(lx[1:], ly[1:]):
        if x - prev_x != 1 or y != prev_y:
            if cur_len >= min_len:
                runs.append((1 + prev_x.item() - cur_len, prev_y.item(), cur_len))
            cur_len = 1
        else:
            cur_len += 1

        prev_x = x
        prev_y = y

    if cur_len >= min_len:
        runs.append((1 + prev_x.item() - cur_len, prev_y.item(), cur_len))

    return runs

# --------------------

# ra and rb are 1-based indexes of the room
def candidate_wall_runs(
    face_walls,
    room_mask,
    inv_room_mask,
    ra, rb,
):

    # Keep ra > rb
    if ra < rb:
        ra, rb = rb, ra
        
    all_runs = []

    ra_mask = (room_mask == ra)

    for i, fw in enumerate(face_walls):
        ws = (fw * ra_mask).byte()
        ws = _restrict_touching(inv_room_mask, room_mask, ws, ra, rb)
        lx, ly = torch.where(ws > 0)
    
        if len(lx) == 0:
            continue
    
        transpose = (i == 0 or i == 2)
        orient = 'h' if transpose else 'v'
    
        if transpose:
            lx, ly = ly, lx
    
        runs = _extract_walls_runs(lx, ly)

        if transpose:
            all_runs.extend(
                (y, x, len, orient)
                for (x, y, len) in runs
            )
        else:
            all_runs.extend(
                (x, y, len, orient)
                for (x, y, len) in runs
            )

    return all_runs


# ---------------------

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