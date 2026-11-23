import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from General_Functions import General_Functions

torch.set_printoptions(profile="full")

matrix = torch.arange(25).reshape(5, 5) # NxN type tensor, value at [i, j] is 5*i + j
indexes = torch.tensor([3, 0, 2]) # n, type tensor (like the batch indexes)

sub_matrix = General_Functions().make_sub_matrix(matrix, indexes)

print("Matrix:", matrix)
print("Indexes:", indexes)
print(sub_matrix)

labels = torch.tensor([0, 1, 0, 2, 1]) # labels of the whole dataset
constraint_matrix = General_Functions().make_constraint_matrix(labels)

sub_matrix = General_Functions().make_sub_matrix(constraint_matrix, indexes)

print("Labels:", labels)
print("Indexes:", indexes)
print(constraint_matrix)
print(sub_matrix)
