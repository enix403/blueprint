import torch

from minimal.walls import (
    CC_TL,
    CC_TR,
    CC_BR,
    CC_BL,
    CC_T,
    CC_R,
    CC_B,
    CC_L
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


def candidate_walls(
    face_walls,
    room_mask,
    ra, rb
):
    if ra < rb:
        ra, rb = rb, ra
    for i, fw in enumerate(face_walls):
        ws = fw.clone()
        ws[room_mask != ra] = 0
        ws = restrict_touching(room_mask, ws, ra, rb)

        loc_x, loc_y = torch.where(ws > 0)

def restrict_touching(room_mask, ra_walls, ra, rb):
    room_mask = room_mask + 1
    ra = ra + 1
    rb = rb + 1
    
    room_mask[room_mask == ra] = 0
    room_mask[room_mask == rb] = 0

    restricted = (_conv_mask(room_mask, _res_kernel) == 0).byte()
    restricted[ra_walls == 0] = 0

    return restricted


_res_kernel = torch.tensor([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
], dtype=torch.int8)


def extract_walls_runs(lx, ly, min_len: int=4):
    runs = []

    prev_x = lx[0]
    prev_y = ly[0]
    cur_len = 1

    for x, y in zip(lx[1:], ly[1:]):
        if x - prev_x != 1 or y != prev_y:
            if cur_len >= min_len:
                runs.append((x.item() - cur_len, cur_len))
            cur_len = 1
        else:
            cur_len += 1

        prev_x = x
        prev_y = y

    if cur_len >= min_len:
        runs.append((prev_x.item() - cur_len, cur_len))

    return runs

