import torch

# Create a tensor of size 2x2
a = torch.tensor([[1, 2], [3, 4]])

# Create another tensor of size 2x2
b = torch.tensor([[5, 6], [7, 8]])

# Perform element-wise addition of the two tensors
c = a + b

# Print the result
print(c)
