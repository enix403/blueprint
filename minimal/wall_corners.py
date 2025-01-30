import torch


kern1 = torch.tensor([
    [-1,  2,  0],
    [ 2, -1,  0],
    [ 0,  0,  0]
], dtype=torch.int8)

all_kerns1 = [
    kern1,
    torch.flip(kern1, (0,)),
    torch.flip(kern1, (1,)),
    torch.flip(kern1, (0, 1)),
]

# =====================

kern2 = torch.tensor([
    [ 2,  0,  0],
    [ 2,  2,  0],
    [ 2,  2,  2]
], dtype=torch.int8)

all_kerns2 = [
    kern2,
    torch.flip(kern2, (0,)),
    torch.flip(kern2, (1,)),
    torch.flip(kern2, (0, 1)),
]

# =====================
