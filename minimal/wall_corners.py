df = torch.tensor([
    [2, -1],
    [-1, 2],
], dtype=torch.int8)


kern1 = torch.zeros((5, 5), dtype=torch.int8)
kern2 = torch.zeros((5, 5), dtype=torch.int8)

kern1[1:3,2:4] = df
kern2[2:4,1:3] = df

print(kern1)
print()
print(kern2)