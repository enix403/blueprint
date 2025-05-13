import random
import torch

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


def _remove_inner_holes(mask: torch.tensor):
    # TODO: implement
    pass

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
            outside_mask = mask
        else:
            outside_mask += mask

    outside_mask = 1 - outside_mask

    # A plan may have some "holes" in its interior.
    # These must be removed since a frontdoor cannot
    # be placed there
    _remove_inner_holes(outside_mask)

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
