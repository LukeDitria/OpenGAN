import torch
import numpy as np

def find_neighbours(num_neighbours, centres, queries=None):

	if queries is None:
		dist_matrix = torch.cdist(centres, centres)
		dist_matrix[torch.diag(torch.ones(dist_matrix.size(0))).type(torch.bool)] = float('inf')
	else:
		dist_matrix = torch.cdist(queries, centres)

	return torch.argsort(dist_matrix)[:,0:num_neighbours]

	