import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from General_Functions import General_Functions
from Objectives import Objectives

torch.set_printoptions(profile="full")

labels = torch.tensor([0, 1, 0, 2, 1]) # labels of the whole dataset
constraint_matrix = General_Functions().make_constraint_matrix(labels)

indexes = torch.tensor([3, 0, 2]) # n, type tensor (like the batch indexes)
code = torch.tensor([[0., 0.], [3., 4.], [6., 8.]]) # nxd type tensor (like the encoder output)

distance_matrix = General_Functions().make_distance_matrix(code)
sub_constraint_matrix = General_Functions().make_sub_matrix(constraint_matrix, indexes)

loss = Objectives('cpu').semi_supervised_loss(distance_matrix, sub_constraint_matrix)

print("Distance matrix:", distance_matrix)
print("Sub constraint matrix:", sub_constraint_matrix)
print("Loss:", loss) # (0 - 5 - 10 - 5 + 0 + 5 - 10 + 5 + 0) / 3 = -20 / 3 = -6.6667
