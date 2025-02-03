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


def candidate_wall_runs(
    face_walls,
    room_masks,
    ra, rb
):
    # Keep ra > rb
    if ra < rb:
        ra, rb = rb, ra
        
    all_runs = []
    
    for i, fw in enumerate(face_walls):
        ws = fw.clone()
        ws[room_masks != ra] = 0
        ws = restrict_touching(room_masks, ws, ra, rb)
        lx, ly = torch.where(ws > 0)
    
        if len(lx) == 0:
            continue
    
        transpose = (i == 0 or i == 3)
        orient = 'h' if transpose else 'v'
    
        if transpose:
            lx, ly = ly, lx
    
        runs = extract_walls_runs(lx, ly)
    
        if transpose:
            all_runs.extend(
                (y, x, len)
                for (x, y, len) in runs
            )
        else:
            all_runs.extend(
                (x, y, len, orient)
                for (x, y, len) in runs
            )

    return all_runs