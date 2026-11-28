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

# Easy numbers to verify by hand
labels = torch.tensor([0, 0, 1, 1]) # points 0 and 1 have the same label, point 2 a different one
constraint_matrix = General_Functions().make_constraint_matrix(labels, optimizer_version=True)

indexes = torch.tensor([0, 1, 2]) # batch of the first 3 points
code = torch.tensor([[0.], [1.], [4.]]) # 1-d codes, so each distance is just |a - b|

distance_matrix = General_Functions().make_distance_matrix(code)
# [[0, 1, 4],
#  [1, 0, 3],
#  [4, 3, 0]]

sub_constraint_matrix = General_Functions().make_sub_matrix(constraint_matrix, indexes)
# [[ 0,  1, -1],
#  [ 1,  0, -1],
#  [-1, -1,  0]]

loss = Objectives('cpu').semi_supervised_loss(distance_matrix, sub_constraint_matrix)
# row 0: 0*0 + 1*1 + 4*(-1) = -3
# row 1: 1*1 + 0*0 + 3*(-1) = -2
# row 2: 4*(-1) + 3*(-1) + 0*0 = -7
# sum = -12, n = 3 -> loss = -12 / 3 = -4

print("Distance matrix:", distance_matrix)
print("Sub constraint matrix:", sub_constraint_matrix)
print("Loss:", loss)
