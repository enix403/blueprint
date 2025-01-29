def up(m):
    """Shifts a 2D mask `m` up"""
    top, rest = m[:1, :], m[1:, :]
    return torch.cat([rest, top], dim=0)

def down(m):
    """Shifts a 2D mask `m` down"""
    rest, bottom = m[:-1, :], m[-1:, :]
    return torch.cat([bottom, rest], dim=0)

def left(m):
    """Shifts a 2D mask `m` left"""
    first, rest = m[:, :1], m[:, 1:]
    return torch.cat([rest, first], dim=1)

def right(m):
    """Shifts a 2D mask `m` down"""
    rest, last = m[:, :-1], m[:, -1:]
    return torch.cat([last, rest], dim=1)

# -----

def walls_between(room, check_room):
    e_top = down((up(room) + check_room == 2).byte())
    e_bottom = up((down(room) + check_room == 2).byte())

    e_left = right((left(room) + check_room == 2).byte())
    e_right = left((right(room) + check_room == 2).byte())

    return (e_top + e_right + e_left + e_bottom).clamp_(0, 1)


# inner_mask = torch.zeros_like(masks[0])
# inner_walls = torch.zeros_like(masks[0])

# for i in range(len(masks)):
#     inner_mask += masks[i]
#     for j in range(i + 1, len(masks)):
#         inner_walls += walls_between(masks[i], masks[j])

# inner_mask.clamp_(0, 1)
# inner_walls.clamp_(0, 1)

# outer_mask = (1 - inner_mask).byte()
# outer_walls = walls_between(inner_mask, outer_mask)

# all_walls = (inner_walls + outer_walls).clamp_(0, 1)

# -----


def disconnections(grid):
    grid = grid.float().unsqueeze(0).unsqueeze(0)
    
    # Define 2x2 kernels for diagonal patterns
    kernel1 = torch.tensor([[2, -1], [-1, 2]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel2 = torch.tensor([[-1, 2], [2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Apply convolution
    output1 = F.conv2d(grid, kernel1)
    output2 = F.conv2d(grid, kernel2)

    m1 = (output1 == 4).byte().squeeze()
    m2 = (output2 == 4).byte().squeeze()

    total = torch.zeros_like(grid)

    total[:-1, :-1] += m1
    total[1:, 1:] += m1

    total[:-1, 1:] += m2
    total[1:, :-1] += m2

    total.clamp_max_(1)

    return total
"""

walls can be added at low res and scaled as needed
walls need to keep orientation
walls need to know which rect/room they belong to
"join" disconnected walls
make sure extra "joined" bits of walls are properly
    connected to representation

"""