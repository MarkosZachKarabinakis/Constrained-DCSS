import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from General_Functions import General_Functions

torch.set_printoptions(profile="full")

labels = torch.tensor([[0], [1], [2], [3], [4], [0], [2], [4]]) # nx1 type tensor

constraint_matrix = General_Functions().make_constraint_matrix(labels)

print("Labels:", labels.flatten())
print(constraint_matrix)

labels = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4]) # n, type tensor

constraint_matrix = General_Functions().make_constraint_matrix(labels)

print("Labels:", labels)
print(constraint_matrix)
