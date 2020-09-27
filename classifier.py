import torch
from neighbours import find_neighbours

class GaussianKernels(torch.nn.Module):
    def __init__(self, num_classes, num_neighbours, num_centres, sigma):
        torch.nn.Module.__init__(self)

        self.num_classes       = num_classes
        self.num_neighbours    = num_neighbours
        self.gaussian_constant = 1.0/(2.0 * (sigma**2))

        #Trainable per-kernel weights - parameterised in log-space (exp(0) = 1)
        self.weight = torch.nn.Parameter(torch.zeros(num_centres, requires_grad=True))
        
    def forward(self, features, centres, centre_labels, neighbours=None):

        batch_size_it  = features.size(0)
        num_dims       = features.size(1)
        p      = []
        
        #Tensor with [0,1,...,num_classes]
        classes = torch.linspace(0,self.num_classes-1,self.num_classes).type(torch.LongTensor).view(self.num_classes,1).to(features.device)

        #Per-kernel weights (self.weight in log space)
        kernel_weights = torch.exp(self.weight)

        #If in test phase, find neighbours
        if neighbours is None:
            neighbours = find_neighbours(self.num_neighbours, centres, features)
        
        for ii in range(batch_size_it):
            
            #Neighbour indices for current example
            neighbours_ii = neighbours[ii,:]
            
            #Squared Euclidean distance to neighbours
            d = torch.pow( features[ii,:] - centres[neighbours_ii], 2 ).sum(1)
            
            #Weighted Gaussian distance
            d = torch.exp( -d * self.gaussian_constant ) * kernel_weights[neighbours_ii]

            #Labels of neighbouring centres
            neighbour_labels = centre_labels[neighbours_ii].view(1,neighbours_ii.size(0))

            #Sum per-class influence of neighbouring centres (avoiding loops - need 2D tensors)
            p_arr      = torch.zeros(self.num_classes,d.size(0)).type(torch.FloatTensor).to(features.device)
            idx        = classes==neighbour_labels
            p_arr[idx] = d.expand(self.num_classes,-1)[idx]
            p_ii       = p_arr.sum(1)                         #Unnormalsied class probability distribution

            #Avoid divide by zero and log(0)
            p_ii[p_ii==0] = 1e-10

            #Normalise
            p_ii = p_ii / p_ii.sum()

            #Convert to log-prob
            p.append(torch.log(p_ii).view(1,self.num_classes))
        
        return torch.cat(p)
