import random
import torch

from minimal.common import conv_mask

DOOR_LENGTH = 10

# --------------------


def _extract_walls_runs(lx, ly, min_len: int = DOOR_LENGTH):

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

_res_kernel = torch.tensor(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=torch.int8,
)


# Keep walls that connect room ra to rb only and none else
def _restrict_touching(room_masks, ra, rb):
    col_mask = (1 - room_masks[ra]) * (1 - room_masks[rb])
    return (conv_mask(col_mask, _res_kernel) == 0).byte()


# --------------------


# wall runs that can be cut between room ra and rb
# 0 based indexes
def _candidate_wall_runs(face_walls, room_masks, ra, rb):

    # Keep ra > rb
    if ra < rb:
        ra, rb = rb, ra

    all_runs = []

    ra_mask = room_masks[ra]

    for i, fw in enumerate(face_walls):
        ws = fw * ra_mask * _restrict_touching(room_masks, ra, rb)

        # ------

        lx, ly = torch.where(ws > 0)

        if len(lx) == 0:
            continue

        transpose = i == 0 or i == 2
        orient = "h" if transpose else "v"

        if transpose:
            lx, ly = ly, lx

        runs = _extract_walls_runs(lx, ly)

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
    R,
    rooms_to_join,
    room_masks,
    face_walls,
):
    doors = []

    for ra, rb in rooms_to_join:
        cruns = _candidate_wall_runs(face_walls, room_masks, ra - 1, rb - 1)

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
