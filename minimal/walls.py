"""

walls can be added at low res and scaled as needed
walls need to keep orientation
walls need to know which rect/room they belong to
make sure extra "joined" bits of walls are properly
    connected to representation

what about parallel displaced walls? like this
    ----
        ----

"""

ftl = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
    [0,  0, -1,  2,  2],
    [0,  0,  2, -1,  0],
    [0,  0,  2,  0,  0],
], dtype=torch.int8).unsqueeze(0).unsqueeze(0)

ftr = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
    [2,  2, -1,  0,  0],
    [0, -1,  2,  0,  0],
    [0,  0,  2,  0,  0],
], dtype=torch.int8).unsqueeze(0).unsqueeze(0)

fbr = torch.tensor([
    [0,  0,  2,  0,  0],
    [0, -1,  2,  0,  0],
    [2,  2, -1,  0,  0],
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8).unsqueeze(0).unsqueeze(0)

fbl = torch.tensor([
    [0,  0,  2,  0,  0],
    [0,  0,  2, -1,  0],
    [0,  0, -1,  2,  2],
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8).unsqueeze(0).unsqueeze(0)


# ----------------------

fp1 = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
    [2,  2, -1,  0,  0],
    [0, -1,  2,  2,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8).unsqueeze(0).unsqueeze(0)

fp2 = torch.tensor([
    [0, 0,  0,  0,  0],
    [0, 2,  2, -1,  0],
    [0, 0, -1,  2,  2],
    [0, 0,  0,  0,  0],
    [0, 0,  0,  0,  0],
], dtype=torch.int8).unsqueeze(0).unsqueeze(0)

fp3 = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  2,  0],
    [0,  0, -1,  2,  0],
    [0,  0,  2, -1,  0],
    [0,  0,  2,  0,  0],
], dtype=torch.int8).unsqueeze(0).unsqueeze(0)

fp4 = torch.tensor([
    [0,  0,  2,  0,  0],
    [0, -1,  2,  0,  0],
    [0,  2, -1,  0,  0],
    [0,  2,  0,  0,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8).unsqueeze(0).unsqueeze(0)

# ----------------------

def detect_unjoined_corners(walls, inner_mask):
    initial = walls

    walls = walls.to(torch.int8).unsqueeze(0).unsqueeze(0)
    walls = walls.clone()

    for kernel in [ftl, ftr, fbr, fbl]:
        res = F.conv2d(walls, kernel, padding=2)
        res = (res == 8).byte()
        
        walls += res
        walls.clamp_max_(1)

    p_walls = torch.zeros_like(initial).unsqueeze(0).unsqueeze(0)

    for kernel in [fp1, fp2, fp3, fp4]:
        res = F.conv2d(walls, kernel, padding=2)
        res = (res == 8).byte()

        p_walls += res
        p_walls.clamp_max_(1)

    p_walls = torch.logical_and(p_walls, inner_mask)

    walls += p_walls
    walls.clamp_max_(1)

    return walls.squeeze() - initial