import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from General_Functions import General_Functions

torch.set_printoptions(profile="full")

latent = torch.tensor([[0., 0.], [3., 4.], [6., 8.]]) # nxd type tensor (like the encoder output)

distance_matrix = General_Functions().make_distance_matrix(latent)

print("Latent:", latent)
print(distance_matrix)

latent = torch.tensor([[0.], [3.], [5.]]) # nx1 type tensor

distance_matrix = General_Functions().make_distance_matrix(latent)

print("Latent:", latent.flatten())
print(distance_matrix)

latent = torch.tensor([0., 3., 5.]) # n, type tensor

distance_matrix = General_Functions().make_distance_matrix(latent)

print("Latent:", latent)
print(distance_matrix)
