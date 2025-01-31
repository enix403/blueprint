import torch

from minimal.walls import (
    BOUNDARY_TOP,
    BOUNDARY_RIGHT,
    BOUNDARY_BOTTOM,
    BOUNDARY_LEFT
)

def _scale_orientation(
    out_mask,
    orient_mask,
    target_orientation,
    scale_x: int, scale_y: int,
    dup_x: bool, dup_y: bool,
    shift_x: bool, shift_y: bool,
):
    loc_x, loc_y = torch.where(orient_mask & target_orientation > 0)

    loc_x *= scale_x
    loc_y *= scale_y

    loc_x = torch.cat([loc_x + i * int(dup_x) for i in range(scale_x)])
    loc_y = torch.cat([loc_y + i * int(dup_y) for i in range(scale_y)])

    loc_x += (scale_x - 1) * int(shift_x)
    loc_y += (scale_y - 1) * int(shift_y)

    out_mask[loc_x, loc_y] = 1


def scale_walls(orient_mask, scale_x, scale_y):

    scaled_walls_mask = torch.zeros((64*scale_x, 64*scale_y), dtype=torch.uint8)

    _scale_orientation(scaled_walls_mask, orient_mask, BOUNDARY_TOP, scale_x, scale_y, 0, 1, 0, 0)
    _scale_orientation(scaled_walls_mask, orient_mask, BOUNDARY_RIGHT, scale_x, scale_y, 1, 0, 0, 1)
    _scale_orientation(scaled_walls_mask, orient_mask, BOUNDARY_BOTTOM, scale_x, scale_y, 0, 1, 1, 0)
    _scale_orientation(scaled_walls_mask, orient_mask, BOUNDARY_LEFT, scale_x, scale_y, 1, 0, 0, 0)

    scale_corners(orient_mask, scale_x, scale_y, scaled_walls_mask)

    return scaled_walls_mask

# ------------------------------------

def scale_corners(
    orient_mask,
    scale_x, scale_y,
    out_mask
):
    for x, y in zip(*torch.where(orient_mask == 16)):
        for name, dv, dh in _neighbours:
            nv = (x + dv[0], y + dv[1])
            nh = (x + dh[0], y + dh[1])

            v_val = extract_v(orient_mask, nv, name[0])
            h_val = extract_h(orient_mask, nh, name[1])

            if not v_val or not h_val:
                continue

            conf = _orient_to_name[h_val.item()] + _orient_to_name[v_val.item()]

            mask = create_cell_mask(_all_confs[name][conf], (scale_x, scale_y))

            xp = x * scale_x
            yp = y * scale_y

            out_mask[xp:xp + scale_x, yp:yp + scale_y] = mask
            

def extract_v(orient_mask, loc, prio):
    val = orient_mask[loc[0], loc[1]]

    t_val = val & BOUNDARY_TOP
    b_val = val & BOUNDARY_BOTTOM

    if prio == "t":
        return t_val or b_val
    else:
        return b_val or t_val

def extract_h(orient_mask, loc, prio):
    val = orient_mask[loc[0], loc[1]]

    r_val = val & BOUNDARY_RIGHT
    l_val = val & BOUNDARY_LEFT

    if prio == "r":
        return r_val or l_val
    else:
        return l_val or r_val

_neighbours = [
    ("tl", (0, -1), (-1, 0)),
    ("tr", (0,  1), (-1, 0)),
    ("br", (0,  1), ( 1, 0)),
    ("bl", (0, -1), ( 1, 0)),
]

_orient_to_name = {
    BOUNDARY_TOP: 't',
    BOUNDARY_RIGHT: 'r',
    BOUNDARY_BOTTOM: 'b',
    BOUNDARY_LEFT: 'l',
}

c1 =  1
c2 =  2
c3 =  3
c4 =  4
e1 =  5
e2 =  6
e3 =  7
e4 =  8
d1 =  9
d2 = 10
d3 = 11
d4 = 12

_all_confs = {
    "tl": { "lt": c1, "lb": e4, "rt": e1, "rb": d3 },
    "tr": { "lt": e1, "lb": d4, "rt": c2, "rb": e2 },
    "br": { "lt": d1, "lb": e3, "rt": e2, "rb": c3 },
    "bl": { "lt": e4, "lb": c4, "rt": d2, "rb": e3 },
}

def create_cell_mask(conf, shape):
    out = torch.zeros(shape, dtype=torch.uint8)

    if   conf == c1: out[ 0,  0] = 1
    elif conf == c2: out[ 0, -1] = 1
    elif conf == c3: out[-1, -1] = 1
    elif conf == c4: out[-1,  0] = 1

    elif conf == e1: out[ 0,  :] = 1
    elif conf == e2: out[ :, -1] = 1
    elif conf == e3: out[-1,  :] = 1
    elif conf == e4: out[ :,  0] = 1

    elif conf == d1: out[ :,  0], out[ 0,  :] = 1, 1
    elif conf == d2: out[ 0,  :], out[ :, -1] = 1, 1
    elif conf == d3: out[ :, -1], out[-1,  :] = 1, 1
    elif conf == d4: out[-1,  :], out[ :,  0] = 1, 1

    return out


