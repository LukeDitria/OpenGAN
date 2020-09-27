
# coding: utf-8
#Pytorch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.datasets as Datasets
from   torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models

#Python
import numpy as np
import random
import os
import time
import argparse

#Custom
from gan import Generator256 as Generator
from gan import Discriminator256 as Discriminator
from loader import MultiFolderLoader

parser = argparse.ArgumentParser(description="Train FEC-GAN model.")

parser.add_argument("--save_name", help="Experiment prefix", required=True)
parser.add_argument("--save_dir", help="Save directory location", required=True)
parser.add_argument("--data_dir", help="Root dataset directory", required=True)
parser.add_argument("--fe_model", help="Path to feature extractor weights", required=True)

parser.add_argument('--gpu_ids', nargs='+', help="List of gpu indexs", type=int, default = [0])
parser.add_argument("--batch_size", help="Batch size", type=int, default=48)
parser.add_argument("--noise_dem", help="Latent vector length", type=int, default=128)
parser.add_argument("--feature_dem", help="Feature vector length (will depend on feature extractor)", type=int, default=512)
parser.add_argument("--load_checkpoint", help="Load checkpoint or start from scratch")
parser.add_argument("--num_classes", help="Number of training (known) classes", type=int, default=82)
parser.add_argument("--val_classes", help="Number of validation (novel) classes", type=int, default=20)
parser.add_argument("--ch_multi", help="Channel multiplier", type=int, default=32)
parser.add_argument("--norm_type", help="Conditional normalisation type: batch (bn) or instance (in)", type=str, default="bn")
parser.add_argument("--disc_steps", help="Number of discriminator steps per generator steps", type=int, default=1)
parser.add_argument("--f_loss_scale", help="Scaling of feature loss", type=float, default=0.01)
parser.add_argument("--dlr", help="Discriminator learning rate", type=float, default=1e-4)
parser.add_argument("--glr", help="Generator learning rate", type=float, default=1e-4)
parser.add_argument("--num_of_iters", help="Number of training iterations", type=int, default=60000)
parser.add_argument("--iters_per_sample", help="Iterations per random sample save", type=int, default=50)
parser.add_argument("--iters_per_save", help="Iterations per model and fixed input save", type=int, default=500)
parser.add_argument("--im_ext", default="jpg", help="Dataset image file extensions (e.g. jpg, png)", type=str)

args = parser.parse_args()

#dataloader Params
image_size = 256

#Feature loss set as MSE
feature_loss_function = nn.MSELoss()

#Checkpoint Params
start_epoch  = 0

model_name = args.save_name + "_" + str(image_size) + "_lr_"  + str(round(args.dlr/args.glr))  + "_BS_" + str (args.batch_size)

#check if a cuda enabled gpu is available
if not torch.cuda.is_available():
    raise ValueError("Cuda Not Avaliable")

device = torch.device(args.gpu_ids[0])
torch.cuda.set_device(args.gpu_ids[0])

print("devices to use")
for device_indx in args.gpu_ids:
    print(device_indx)
    
#transform to apply on dataset
data_transform  = T.Compose([
                               T.Resize(image_size),
                               T.CenterCrop(image_size),
                               T.RandomHorizontalFlip(p=0.5),
                               T.ToTensor(),
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Create known (train) and novel datasets
trainset = MultiFolderLoader(root=args.data_dir, transform=data_transform, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=False)
novelset = MultiFolderLoader(root=args.data_dir, transform=data_transform, num_classes = args.val_classes, start_indx = args.num_classes, img_type = "."+args.im_ext, ret_class=False)

#Load datasets into dataloaders
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory = True)
novelloader = DataLoader(novelset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory = True)

#Calculate rough number of epochs needed to complete training itterations
nepoch = (args.num_of_iters)//len(trainloader)

#Obtain and view images from novel classes
dataiter = iter(novelloader)
novel_images = dataiter.next()
out = vutils.make_grid(novel_images[0:8])

#Helper function to approximate training time left
def time_left(start_time, current_time, itters_past, max_itters):
    seconds_left = ((current_time - start_time)/(itters_past+1e-6))*(max_itters - itters_past)
    return seconds_left/3600

#Create and load the metric feature extractor network
feature_extractor_network = models.resnet18(pretrained=False)
modules = list(feature_extractor_network.children())[:-1] #Remove fully connected layer
modules.append(nn.Flatten())
feature_extractor_network = nn.Sequential(*modules)
feature_extractor_network = feature_extractor_network.to(device)
feature_extractor_network.load_state_dict(torch.load(args.fe_model, map_location=device))
feature_extractor_network = feature_extractor_network.eval()

#Setup networks
generator_network = Generator(in_noise = args.noise_dem, in_features = args.feature_dem, ch = args.ch_multi, norm_type = args.norm_type).to(device)
discriminator_network = Discriminator(channels = novel_images.shape[1], in_features = args.feature_dem, ch = args.ch_multi, norm_type = args.norm_type).to(device)

#Setup optimizers
d_optimizer = optim.Adam(discriminator_network.parameters(), lr=args.dlr, betas=(0.0, 0.999))
g_optimizer = optim.Adam(generator_network.parameters(), lr=args.glr, betas=(0.0, 0.999))

#Create the save directory if it does note exist
if not os.path.isdir(args.save_dir + "/Models"):
    os.makedirs(args.save_dir + "/Models")
if not os.path.isdir(args.save_dir + "/Samples"):
    os.makedirs(args.save_dir + "/Samples")
    
if args.load_checkpoint:
    checkpoint = torch.load(args.save_dir + "/Models/" + model_name + ".pt", map_location = "cpu")
    print("Checkpoint loaded")
    generator_network.load_state_dict(checkpoint['G_state_dict'])
    discriminator_network.load_state_dict(checkpoint['D_state_dict'])
    g_optimizer.load_state_dict(checkpoint['optimizerG_state_dict'])
    d_optimizer.load_state_dict(checkpoint['optimizerD_state_dict'])
    start_epoch = checkpoint["epoch"]
    #If checkpoint does exist and Start_From_Checkpoint = False
    #Raise an error to prevent accidental overwriting
else:
    if os.path.isfile(args.save_dir + "/Models/" + model_name + ".pt"):
        raise ValueError("Warning Checkpoint exists")
    else:
        print("Starting from scratch")


#Wrap networks with DataParallel to train on multiple GPUs
generator_network = nn.DataParallel(generator_network.to(device), device_ids = args.gpu_ids)
discriminator_network = nn.DataParallel(discriminator_network.to(device), device_ids = args.gpu_ids)
feature_extractor_network = nn.DataParallel(feature_extractor_network, device_ids = args.gpu_ids)

#Extract Features of novel class examples for validation
with torch.no_grad():
    fixed_features = feature_extractor_network(novel_images)
    fixed_features = fixed_features.detach()
    
#Fixed latent noise vector for validation
fixed_noise = torch.randn(args.batch_size, args.noise_dem)

#Save the real images of the novel classes to compare
vutils.save_image(novel_images,"%s_real_samples.png" % (args.save_dir + "/Samples/" + model_name), normalize=True)

# A few checks
with torch.no_grad():
    if not(novel_images.shape[2] == image_size):
        raise ValueError('Image size invalid')

    if not(fixed_features.shape[1] == args.feature_dem):
        raise ValueError('Feature size invalid')

    if not(generator_network(fixed_noise, fixed_features).shape[2] == image_size):
        raise ValueError('Generator size invalid')

    if not(discriminator_network(novel_images, fixed_features).shape[1] == 1):
        raise ValueError('Discriminator output invalid')

start_time = time.time()
for epoch in range(start_epoch, nepoch):
    for i, real_images in enumerate(trainloader, 0):
        #Keeping track of the current itteration
        itter = epoch * len(trainloader) + i               
        
        #Get the size of the minibatch
        mini_batch_size = real_images.size(0)
        
        #Extract the metric features of the current minibatch of real images
        #we don't need gradient information for this - features only used as a label
        with torch.no_grad():
            real_features = feature_extractor_network(real_images)

        #Generate fake images using the features of the real images
        noise = torch.randn(mini_batch_size, args.noise_dem)
        fake_images = generator_network(noise, real_features)
        
        
        ##################### Train Discriminator #######################
        for _ in range(args.disc_steps):
            #Pass real images to discriminator conditioned on real features
            d_out_real = discriminator_network(real_images, real_features)
            
            #Pass fake images to discriminator conditioned on real features
            d_out_fake = discriminator_network(fake_images.detach(), real_features)
            
            #Hinge loss for the discriminator with real and fake image outputs
            d_fake_loss = torch.mean(F.relu(1. + d_out_fake))
            d_real_loss = torch.mean(F.relu(1. - d_out_real))
                
            #Combine losses
            d_loss = d_fake_loss + d_real_loss
            
            #Backpropagate gradients through discriminator and take an optimisation step on the discriminator
            discriminator_network.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
        ############################
        
        ##################### Train Generator #######################
        
        #Pass fake images through the discriminator conditioned on real features
        d_out_fake_g = discriminator_network(fake_images, real_features)
        
        #Pass fake images through the metric feature extractor and get the features
        fake_features = feature_extractor_network(fake_images)

        #Calculate the feature loss (MSE)
        feature_loss =  feature_loss_function(fake_features, real_features)
        
        #Use the output of the discriminator to calculate the loss for the generator
        g_loss = -torch.mean(d_out_fake_g)

        #Scale the feature loss and add the GAN loss
        total_g_loss = g_loss + args.f_loss_scale * feature_loss
        
        #Backpropagate gradients through discriminator, feature extractor and generator
        #and take an optimisation step on the generator
        generator_network.zero_grad()
        total_g_loss.backward()
        g_optimizer.step()
        
        with torch.no_grad():
            #Print relivant information
            print('[%d/%d][%d/%d] Dout R/F: [%.2f/%.2f] LossD R/F: [%.2f/%.2f] LossG: %.2f, LossF: %.2f, hours left: %.2f' 
                  % (epoch, nepoch, i, len(trainloader),
                     d_out_real.mean().item() ,d_out_fake.mean().item(), d_real_loss.item() ,d_fake_loss.item(), g_loss.item(), feature_loss.item(), time_left(start_time, time.time(), itter, args.num_of_iters)))

            #Save the current batch of fake images every "itters_per_sample" itteration
            if ((itter+1)%args.iters_per_sample == 0):
                vutils.save_image(fake_images.detach(),
                    "%s_fake_samples_current.png" % (args.save_dir + "/Samples/" + model_name), normalize=True)

            #Save the images generated using the novel images features and fixed noise
            #every "itters_per_save" itteration
            if (itter+1)%args.iters_per_save == 0:
                fake_images = generator_network(fixed_noise, fixed_features)
                vutils.save_image(fake_images.detach(),
                        "%s_fake_samples_epoch_%d.png" % (args.save_dir + "/Samples/" + model_name, epoch), normalize=True)

                #Save a checkpoint every "itters_per_save"
                torch.save({
                            'epoch'                          : epoch,
                            'G_state_dict'                   : generator_network.module.state_dict(),
                            'D_state_dict'                   : discriminator_network.module.state_dict(),
                            'optimizerG_state_dict'          : g_optimizer.state_dict(),
                            'optimizerD_state_dict'          : d_optimizer.state_dict()

                             }, args.save_dir + "/Models/" + model_name + ".pt")         
                

