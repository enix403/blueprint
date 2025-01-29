"""

walls can be added at low res and scaled as needed
walls need to keep orientation
walls need to know which rect/room they belong to
make sure extra "joined" bits of walls are properly
    connected to representation

"""

def detect_unjoined_corners(grid):

    grid = grid.float().unsqueeze(0).unsqueeze(0)

    fbl = torch.tensor([
        [0,  0,  2,  0,  0],
        [0,  0,  2, -1,  0],
        [0,  0, -1,  2,  2],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


    res = F.conv2d(grid, fbl, padding=2)
    res = (res == 8).byte()

    res = res.squeeze()

    return res
