import torch


# disonnected corners
_kerns1 = torch.tensor([
    [-1,  2,  0],
    [ 2, -1,  0],
    [ 0,  0,  0]
], dtype=torch.int8)

all_kerns1 = [
    _kerns1,
    torch.flip(_kerns1, (0,)),
    torch.flip(_kerns1, (1,)),
    torch.flip(_kerns1, (0, 1)),
]

# =====================

# extra corners
_kerns2 = torch.tensor([
    [ 2,  0,  0],
    [ 2,  2,  0],
    [ 2,  2,  2]
], dtype=torch.int8)

all_kerns2 = [
    _kerns2,
    torch.flip(_kerns2, (0,)),
    torch.flip(_kerns2, (1,)),
    torch.flip(_kerns2, (0, 1)),
]

# =====================

# L shaped corner for orientation masking

_kerns3 = torch.tensor([
    [ 0,  1,  0],
    [ 1,  0,  0],
    [ 0,  0,  0]
], dtype=torch.int8)

all_kerns3 = [
    _kerns3, # R + B
    torch.flip(_kerns3, (0,)), # T + R
    torch.flip(_kerns3, (1,)), # B + L
    torch.flip(_kerns3, (0, 1)), # T + L
]


