"""

walls can be added at low res and scaled as needed
walls need to keep orientation
walls need to know which rect/room they belong to
make sure extra "joined" bits of walls are properly
    connected to representation

"""

# def apply_corner_filter(grid, filter):

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

def detect_unjoined_corners(grid):
    edges = torch.zeros_like(grid)

    grid = grid.to(torch.int8).unsqueeze(0).unsqueeze(0)

    for kernel in [ftl, ftr, fbr, fbl]:
        res = F.conv2d(grid, kernel, padding=2)
        res = (res == 8).byte()
        res = res.squeeze()

        edges += res

    return edges.clamp_max_(0)
