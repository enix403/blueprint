from typing import List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minimal.rooms import RoomAreas

def _shift_up(m):
    """Shifts a 2D mask `m` up"""
    top, rest = m[:1, :], m[1:, :]
    return torch.cat([rest, top], dim=0)

def _shift_down(m):
    """Shifts a 2D mask `m` down"""
    rest, bottom = m[:-1, :], m[-1:, :]
    return torch.cat([bottom, rest], dim=0)

def _shift_left(m):
    """Shifts a 2D mask `m` left"""
    first, rest = m[:, :1], m[:, 1:]
    return torch.cat([rest, first], dim=1)

def _shift_right(m):
    """Shifts a 2D mask `m` down"""
    rest, last = m[:, :-1], m[:, -1:]
    return torch.cat([last, rest], dim=1)


def _conv_mask(mask, kernel, threshold_match = None):
    mask = mask.to(torch.int8).unsqueeze(0).unsqueeze(0)
    # kernel is assumed to be in int8
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    f = kernel.shape[-1]
    padding = (f - 1) // 2

    result = F.conv2d(mask, kernel, padding=padding)
    result = result[0, 0, :, :]

    if threshold_match is not None:
        result = (result == threshold_match).byte()

    return result

# --------------

# fmt: off
BOUNDARY_TOP    = 0b0001
BOUNDARY_RIGHT  = 0b0010
BOUNDARY_BOTTOM = 0b0100
BOUNDARY_LEFT   = 0b1000
# fmt: on

def walls_between(room_mask, check_room_mask):
    a = room_mask
    b = check_room_mask

    e_top = _shift_down((_shift_up(a) + b == 2).byte())
    e_bottom = _shift_up((_shift_down(a) + b == 2).byte())

    e_left = _shift_right((_shift_left(a) + b == 2).byte())
    e_right = _shift_left((_shift_right(a) + b == 2).byte())

    walls_mask = (e_top + e_right + e_bottom + e_left).clamp_max_(1)

    orient_mask = torch.zeros_like(room_mask)

    orient_mask[torch.where(e_top == 1)] += BOUNDARY_TOP
    orient_mask[torch.where(e_right == 1)] += BOUNDARY_RIGHT
    orient_mask[torch.where(e_bottom == 1)] += BOUNDARY_BOTTOM
    orient_mask[torch.where(e_left == 1)] += BOUNDARY_LEFT

    return walls_mask, orient_mask


def intersect_rooms(rooms: list[RoomAreas]):

    # TODO: need a better metric then total_area
    rooms = sorted(rooms, key=lambda r: r.total_area(), reverse=True)
    masks = [room.to_mask() for room in rooms]

    inner_mask = torch.zeros_like(masks[0])
    inner_walls = torch.zeros_like(masks[0])

    orient_mask = torch.zeros_like(inner_walls)

    for i in range(len(masks)):
        inner_mask += masks[i]
        for j in range(i + 1, len(masks)):
            walls_mask_room, orient_mask_room = walls_between(masks[i], masks[j])

            inner_walls += walls_mask_room
            orient_mask += orient_mask_room

    inner_mask.clamp_max_(1)
    inner_walls.clamp_max_(1)

    outer_mask = (1 - inner_mask).byte()
    outer_walls, orient_mask_room = walls_between(inner_mask, outer_mask)
    orient_mask += orient_mask_room

    walls_mask = (inner_walls + outer_walls).clamp_max_(1)

    return walls_mask, orient_mask, inner_mask

# -----------------

# L-shaped disonnected corners
_l_corners = torch.tensor([
    [-1,  2,  0],
    [ 2, -1,  0],
    [ 0,  0,  0]
], dtype=torch.int8)

kerns_l_corners = [
    _l_corners, # R + B
    torch.flip(_l_corners, (0,)), # T + R
    torch.flip(_l_corners, (1,)), # B + L
    torch.flip(_l_corners, (0, 1)), # T + L
]

orientations_l_corners = [
    # do not change order
    BOUNDARY_RIGHT + BOUNDARY_BOTTOM,
    BOUNDARY_TOP + BOUNDARY_RIGHT,
    BOUNDARY_BOTTOM + BOUNDARY_LEFT,
    BOUNDARY_TOP + BOUNDARY_LEFT,
]

# extra/duplicated corners
_dup_corners = torch.tensor([
    [ 2,  0,  0],
    [ 2,  2,  0],
    [ 2,  2,  2]
], dtype=torch.int8)

kerns_dup_corners = [
    _dup_corners,
    torch.flip(_dup_corners, (0,)),
    torch.flip(_dup_corners, (1,)),
    torch.flip(_dup_corners, (0, 1)),
]

def join_wall_corners(walls_mask, orient_mask, inner_mask):
    initial = walls_mask
    walls_mask = walls_mask.clone()

    # --------------------

    p_walls = torch.zeros_like(initial)
    for kernel in kerns_l_corners:
        res = _conv_mask(walls_mask, kernel, 4)
        p_walls.add_(res).clamp_max_(1)

    p_walls *= inner_mask

    walls_mask.add_(p_walls).clamp_max_(1)

    # --------------------

    extra_walls = torch.zeros_like(initial)
    for kernel in kerns_dup_corners:
        res = _conv_mask(walls_mask, kernel, 12)
        extra_walls.add_(res).clamp_max_(1)

    walls_mask -= extra_walls

    # --------------------

    corners = walls_mask - initial

    corners_orient_mask = torch.zeros_like(orient_mask)

    for kern, ort in zip(kerns_l_corners, orientations_l_corners):
        cur_corners = _conv_mask(initial, kern, 4) * corners
        corners_orient_mask[torch.where(cur_corners == 1)] = ort

    # --------------------

    return walls_mask, orient_mask + corners_orient_mask


# =========================

def extract_walls(rooms: list[RoomAreas]):
    walls_mask, orient_mask, inner_mask = intersect_rooms(rooms)
    walls_mask, orient_mask = join_wall_corners(walls_mask, orient_mask, inner_mask)

    return walls_mask, orient_mask
