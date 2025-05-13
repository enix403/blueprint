import random
import torch

from minimal.common import conv_mask
from minimal.walls import _extract_walls_runs

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

        cruns = _candidate_wall_runs(face_walls, room_a_mask, room_b_mask)

        if len(cruns) == 0:
            print(f"Unable to allocate door between {ra} and {rb}")
            continue

        # print(cruns)
        run = cruns[0]
        door = _select_segment_from_run(run)
        doors.append(door)

    return doors


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
