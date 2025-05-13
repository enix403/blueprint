import random
import torch
import cv2
import numpy as np

from minimal.common import conv_mask
from minimal.walls import _extract_walls_runs
from minimal.layout import InputLayout, NodeType
from minimal.rooms import RectGraph

DOOR_LENGTH = 10

# --------------------


def _extract_walls_runs_for_doors(lx, ly):
    return _extract_walls_runs(lx, ly, min_len=DOOR_LENGTH)

# -------------------


_res_kernel = torch.tensor(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=torch.int8,
)


def _restrict_touching(room_a_mask, room_b_mask):
    """
    For every edge of the two rooms, only selects those edges
    that are touching the other room. Note that this may also
    select some other inner cells of both rooms are well. It
    is recommended to select cells only from one of the given
    rooms (e.g by multiplying with room_a_mask) and also discarding
    non-edge cells (e.g by multiplying with a walls mask)
    """

    # This is a mask containing the area outside both of
    # the rooms
    outside_mask = (1 - room_a_mask) * (1 - room_b_mask)

    # TODO: if it is guranteed that the two rooms are non-overlapping, then
    # the above can be simplified to
    #       outside_mask = 1 - (room_a_mask + room_b_mask)

    return (conv_mask(outside_mask, _res_kernel) == 0).byte()

# ---------------


def _candidate_wall_runs(
    face_walls,
    room_a_mask,
    room_b_mask
) -> list[tuple[int, int, int, str]]:
    """
    Calculates wall runs that can be cut between the two rooms.
    The wall runs are placed on the first room i.e room_a_mask 
    """
    all_runs = []

    for i, fw in enumerate(face_walls):
        ws = fw * room_a_mask * _restrict_touching(room_a_mask, room_b_mask)

        # ------

        lx, ly = torch.where(ws > 0)

        if len(lx) == 0:
            continue

        transpose = i == 0 or i == 2
        orient = "h" if transpose else "v"

        if transpose:
            lx, ly = ly, lx

        runs = _extract_walls_runs_for_doors(lx, ly)

        if transpose:
            all_runs.extend((y, x, len, orient) for (x, y, len) in runs)
        else:
            all_runs.extend((x, y, len, orient) for (x, y, len) in runs)

    return all_runs

# -----------------


def _select_segment_from_run(run):
    x, y, l, orient = run

    max_push = l - DOOR_LENGTH

    push = random.randint(0, max_push)

    if orient == "v":
        x += push
    else:
        y += push

    return (x, y, DOOR_LENGTH, orient)


def create_door(
    face_walls,
    room_a_mask,
    room_b_mask
):
    cruns = _candidate_wall_runs(face_walls, room_a_mask, room_b_mask)

    if len(cruns) == 0:
        return None

    # print(cruns)
    run = cruns[0]
    door = _select_segment_from_run(run)
    return door


def create_doors(
    rooms_to_join,
    room_masks,
    face_walls,
):
    doors = []

    for ra, rb in rooms_to_join:
        if ra < rb:
            ra, rb = rb, ra

        room_a_mask = room_masks[ra - 1]
        room_b_mask = room_masks[rb - 1]

        door = create_door(face_walls, room_a_mask, room_b_mask)

        if door is None:
            print(f"Unable to allocate door between {ra} and {rb}")
            continue

        doors.append(door)

    return doors

# ------------


def _remove_inner_with_cv2(x: torch.Tensor) -> torch.Tensor:
    """
    x: 2D torch.uint8 tensor containing only 0s and 1s
    returns: same shape tensor where only the 1-regions connected to the border remain
    """
    # 1. Move to CPU & numpy
    arr = x.cpu().numpy()

    # 2. Label all 4-connected components in one pass (C speed!)
    #    background=0, labels ∈ {0..num_labels-1}
    num_labels, labels = cv2.connectedComponents(arr, connectivity=4)

    # 3. Find which labels touch any edge
    edge_labels = set()
    edge_labels.update(np.unique(labels[0, :]))      # top row
    edge_labels.update(np.unique(labels[-1, :]))      # bottom row
    edge_labels.update(np.unique(labels[:,  0]))      # left  col
    edge_labels.update(np.unique(labels[:, -1]))      # right col

    # 4. Build mask of “keep” pixels
    keep = np.zeros_like(arr, dtype=np.uint8)
    for lbl in edge_labels:
        if lbl != 0:  # skip background
            keep[labels == lbl] = 1

    # 5. Back to torch (on original device)
    return torch.from_numpy(keep).to(x.device).to(torch.uint8)


def _remove_inner_with_dilation(x: torch.Tensor) -> torch.Tensor:
    """
    x: 2D torch.uint8 tensor of 0/1 on either CPU or CUDA
    returns: same shape torch.uint8 where only border-connected 1s remain
    """
    # mask of “reachable” starts with just the 1s on the border
    reachable = torch.zeros_like(x, dtype=torch.bool)
    H, W = x.shape
    b = reachable

    # mark border 1s
    b[0, :] |= (x[0, :] == 1)
    b[-1, :] |= (x[-1, :] == 1)
    b[:,  0] |= (x[:,  0] == 1)
    b[:, -1] |= (x[:, -1] == 1)

    # 4-neighbour kernel: up/down/left/right
    kernel = torch.tensor([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]],
                          dtype=torch.float32,
                          device=x.device).unsqueeze(0).unsqueeze(0)

    # iterative dilation until no more growth
    prev_nnz = -1
    while True:
        # convolve to see where any neighbour is reachable
        # (we convert bool→float so conv2d will work)
        nbrs = F.conv2d(b.float().unsqueeze(0).unsqueeze(0),
                        kernel, padding=1).squeeze(0).squeeze(0)
        # any location that had at least one reachable neighbour
        # and is itself a 1 in x becomes reachable
        b_new = b | ((nbrs > 0) & (x == 1))

        nnz = int(b_new.sum())
        if nnz == prev_nnz:
            break
        prev_nnz = nnz
        b = b_new

    # mask out all unreachable 1s
    return (b.to(torch.uint8) * x)


def _remove_inner_holes(mask: torch.tensor):
    return _remove_inner_with_cv2(mask)
    # or
    # return _remove_inner_with_dilation(mask)


def create_front_doors(
    face_walls,
    rooms: list[RectGraph],
    room_masks: list[torch.tensor],
    layout: InputLayout
):
    # Get a list of rooms that need a front door
    # TODO: Optimize
    front_rooms_idx_layout = set()
    for i, node_type in enumerate(layout.node_types):
        if node_type == NodeType.FRONT_DOOR:
            # Loop through all the edges (unoptimized)
            # To find all rooms that are connected to
            # this frontdoor
            for edge in layout.edges:
                if edge[0] == i:
                    front_rooms_idx_layout.add(edge[1])
                elif edge[1] == i:
                    front_rooms_idx_layout.add(edge[0])

    # Find the corresponding indices in the `rooms` array
    front_rooms_idx = set()
    for i, room in enumerate(rooms):
        if room.room_node_index in front_rooms_idx_layout:
            front_rooms_idx.add(i)

    # Mask containing the area not spanned by any room
    outside_mask = None
    for mask in room_masks:
        if outside_mask is None:
            outside_mask = mask.clone()
        else:
            outside_mask += mask

    outside_mask = 1 - outside_mask

    # A plan may have some "holes" in its interior.
    # These must be removed since a frontdoor cannot
    # be placed there
    outside_mask = _remove_inner_holes(outside_mask)

    front_doors = []

    for i in front_rooms_idx:
        room_a_mask = room_masks[i]
        door = create_door(face_walls, room_a_mask, outside_mask)
        if door is None:
            print(f"Unable to allocate front door foor room {i}")
            continue
        front_doors.append(door)

    return front_doors

# -----------------
# -----------------
# -----------------
# -----------------
# -----------------


def create_cut_wall_mask(wall_mask, doors: list[tuple]):
    walls = (wall_mask > 0).byte()
    for x1, y1, l, o in doors:
        x2, y2 = x1 + 1, y1 + 1
        if o == "h":
            y2 = y1 + l
        else:
            x2 = x1 + l

        walls[x1:x2, y1:y2] = 0

    return walls
