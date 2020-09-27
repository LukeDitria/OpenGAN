import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as functional

from neighbours import find_neighbours
from classifier import GaussianKernels
from loader import MultiFolderLoader

import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Train Gaussian kernel classifier using Resnet18.")
parser.add_argument("--data_dir", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--save_dir", required=True, type=str, help="Models are saved to this directory.")
parser.add_argument("--num_classes", required=True, type=int, help="Number of training classes to use.")

parser.add_argument("--im_ext", default="jpg", type=str, help="Dataset image file extensions (e.g. jpg, png).")
parser.add_argument("--gpu_id", default=None, type=int, help="GPU ID. CPU is used if not supplied.")
parser.add_argument("--sigma", default=10, type=int, help="Gaussian sigma.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--learning_rate", default=1e-5, type=int, help="learning_rate")
parser.add_argument("--update_interval", default=5, type=int, help="Stored centres/neighbours are updated every update_interval epochs.")
parser.add_argument("--max_epochs", default=15, type=int, help="Maximum training length (epochs).")

args = parser.parse_args()

"""
Configuration
"""

#Data info
input_size = 256
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]

#Resnet18 model
model = torchvision.models.resnet18(pretrained=True)

#Remove fully connected layer
modules = list(model.children())[:-1]
modules.append(nn.Flatten())
model = nn.Sequential(*modules)

kernel_weights_lr = args.learning_rate*1
num_neighbours    = 200
eval_interval     = args.update_interval

#Set GPU ID or 'cpu'
if args.gpu_id is None:
	device = torch.device('cpu')
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
	device = torch.device('cuda:0')

"""
Set up DataLoaders
"""

#Transformations/pre-processing operations
train_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

update_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])


train_dataset  = MultiFolderLoader(args.data_dir, train_transforms, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=True)
update_dataset = MultiFolderLoader(args.data_dir, update_transforms, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=True)

#Data loaders to handle iterating over datasets
train_loader  = DataLoader(train_dataset,  batch_size=args.batch_size, shuffle=True,  num_workers=3)
update_loader = DataLoader(update_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)


"""
Create Gaussian kernel classifier
"""
model = model.to(device)
#model = model.train()
model = model.eval()

def update_centres():

	#Disable dropout, use global stats for batchnorm
	model.eval()

	#Disable learning
	with torch.no_grad():

		#Update stored centres
		for i, data in enumerate(update_loader, 0):

			# Get the inputs; data is a list of [inputs, labels]. Send to GPU
			inputs, labels, indices = data
			inputs = inputs.to(device)

			#Extract features for batch
			extracted_features = model(inputs)

			#Save to centres tensor
			idx = i*args.batch_size
			centres[idx:idx + extracted_features.shape[0], :] = extracted_features

	#model.train()
	model.eval()

	return centres

def save_model():
	torch.save(model.state_dict(), args.save_dir + "/model.pt")
	torch.save(kernel_classifier.state_dict(), args.save_dir + "/classifier.pt")
	torch.save(centres, args.save_dir + "/centres.pt")

num_train = len(update_loader.dataset)

with torch.no_grad():
	num_dims = model(torch.randn(1,3,input_size,input_size).to(device)).size(1)

#Create tensor to store kernel centres
centres = torch.zeros(num_train,num_dims).type(torch.FloatTensor).to(device)
print("Size of centres is {0}".format(centres.size()))

#Create tensor to store labels of centres
centre_labels = torch.LongTensor(update_dataset.get_all_labels()).to(device)

#Create Gaussian kernel classifier
kernel_classifier = GaussianKernels(args.num_classes, num_neighbours, num_train, args.sigma)
kernel_classifier = kernel_classifier.to(device)


"""
Set up loss and optimiser
"""

criterion = nn.NLLLoss()

optimiser = optim.Adam([
                {'params': model.parameters()},
                {'params': kernel_classifier.parameters(), 'lr': kernel_weights_lr}
            ], lr=args.learning_rate)

# exp_lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=step_gamma)


"""
Training
"""
print("Begin training...")
for epoch in range(args.max_epochs):  # loop over the dataset multiple times
	
	#Update stored kernel centres
	if (epoch % args.update_interval) == 0:

		print("Updating kernel centres...")
		centres = update_centres()
		print("Finding training set neighbours...")
		centres = centres.cpu()
		neighbours_tr = find_neighbours( num_neighbours, centres )
		centres = centres.to(device)
		print("Finished update!")

		if epoch > 0:
			save_model()
	
	#Training
	running_loss = 0.0
	running_correct = 0
	for i, data in enumerate(train_loader, 0):
		
		# Get the inputs; data is a list of [inputs, labels]. Send to GPU
		inputs, labels, indices = data
		inputs  = inputs.to(device)
		labels  = labels.to(device).view(-1)
		indices = indices.to(device)

		# Zero the parameter gradients
		optimiser.zero_grad()
		
		log_prob = kernel_classifier( model(inputs), centres, centre_labels, neighbours_tr[indices, :] )

		loss = criterion(log_prob, labels)
		loss.backward()
		optimiser.step()

		running_loss += loss.item()

		#Get the index of the max log-probability
		pred = log_prob.argmax(dim=1, keepdim=True) 
		correct = pred.eq(labels.view_as(pred)).sum().item()
		running_correct += correct

	#Print statistics at end of epoch
	if True:
		print('[{0}, {1:5d}] loss: {2:.3f}, accuracy: {3}/{4} ({5:.4f}%)'.format(
			epoch + 1, i + 1, running_loss / len(train_loader.dataset),
			running_correct, len(train_loader.dataset), 100. * running_correct / len(train_loader.dataset)))
		running_loss = 0.0
		running_correct = 0

	# exp_lr_scheduler.step()

#Update centres final time when done
print("Updating kernel centres (final time)...")
centres = update_centres()
save_model()
