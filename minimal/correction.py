import torch

def largest_rectangle_area(heights):
    """
    Find the largest rectangle area in a histogram.
    
    Args:
        heights (list[int]): A list of heights of histogram bars.

    Returns:
        int: The largest rectangle area.
        int: The start index of the rectangle.
        int: The width of the rectangle.
    """
    stack = []  # Stack to store indices of histogram bars
    max_area = 0
    start_idx = 0
    max_width = 0

    heights.append(0)  # Append a sentinel value to flush the stack
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            if area > max_area:
                max_area = area
                start_idx = stack[-1] + 1 if stack else 0
                max_width = width
        stack.append(i)

    heights.pop()  # Remove the sentinel value
    return max_area, start_idx, max_width

def find_largest_rectangle(grid):
    """
    Find the largest rectangle of 1's in a 2D binary grid using the histogram technique.

    Args:
        grid (torch.Tensor): A 2D binary grid (n x m) of dtype torch.uint8.

    Returns:
        tuple: (x, y, width, height) where x, y are the top-left coordinates,
               width is the width of the rectangle, and height is the height.
    """
    n, m = grid.shape
    heights = torch.zeros(m, dtype=torch.int32)
    max_rectangle = (0, 0, 0, 0)
    max_area = 0

    for i in range(n):
        for j in range(m):
            heights[j] = heights[j] + 1 if grid[i, j] == 1 else 0

        area, start_col, width = largest_rectangle_area(heights.tolist())
        if area > max_area:
            max_area = area
            max_rectangle = (i - area // width + 1, start_col, width, area // width)

    return max_rectangle

def split_into_rectangles(grid):
    """
    Split a binary grid into the smallest set of non-overlapping rectangles.

    Args:
        grid (torch.Tensor): A 2D binary grid (n x m) of dtype torch.uint8.

    Returns:
        list[tuple]: A list of rectangles represented as (x, y, width, height).
    """
    grid = grid.clone()  # Work on a copy to avoid modifying the input
    rectangles = []

    while grid.any():
        x, y, width, height = find_largest_rectangle(grid)
        rectangles.append((x, y, width, height))

        # Set the cells of the found rectangle to 0
        grid[x:x + height, y:y + width] = 0

    return rectangles

# ---------

def reconstruct_mask(room_shape, rects):
    rec = torch.zeros(room_shape, dtype=torch.uint8)

    for x, y, w, h in rects:
        rec[x:x + h, y:y + w] = 1

    return rec

# ---------

# public
def correct_mask(mask):
    rects = split_into_rectangles(room)    


"""
Psuedo code

def CorrectMask(mask):

    # get all the aligned rects of every size
    rects = GetAllAlignedRects(mask)
    polys = MergeIntoPolygons(rects)

    sort polygons by area (largest first)

    # use the largest polygon and discard the rest
    mask = CreateMask(polys[0])
    return mask

def CorrectPlan(plan):
    for mask in plan:
        mask = CorrectMask(mask)

    # TODO: fill overlaps
    # TODO: fill holes

"""