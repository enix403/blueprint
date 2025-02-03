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


# def candidate_walls(
#     face_walls,
#     room_mask,
#     ra, rb
# ):
#     if ra < rb:
#         ra, rb = rb, ra

#     for i, fw in enumerate(face_walls):
#         ws = fw.clone()
#         ws[room_mask != ra] = 0
#         ws = restrict_touching(room_mask, ws, ra, rb)

#         loc_x, loc_y = torch.where(ws > 0)


"""

"""