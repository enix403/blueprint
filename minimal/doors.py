import torch

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
    
        transpose = (i == 0 or i == 3)
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