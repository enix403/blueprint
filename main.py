import os
import numpy as np

import torch
from torchvision.utils import save_image

from minimal.layout import LayoutGraph
from minimal.gen import generate_plan
from minimal import sample_graphs

print("Starting generation")

# ----

g = sample_graphs.two()
plan_masks = generate_plan(g, num_iters=25)

img = plan_masks.render()
img = torch.tensor(np.array(img).transpose((2, 0, 1))) / 255.0
save_image(img, "dump/fp_final_0.png", nrow=1, normalize=False)

# ----

print("Done")