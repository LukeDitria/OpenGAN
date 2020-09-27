import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as Datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import os

class MultiFolderLoader(Dataset):
    def __init__(self, root, transform, num_classes = 150, start_indx = 0, img_type = ".jpg", ret_class = False):
        self.transform   = transform

        self.root = root
        self.ret_class = ret_class
        self.directories = np.sort(os.listdir(root))
        self.annotations = []
        self.class_labels = []
        self.img_type = img_type
        class_label = 0
        print("Loading...")
        print(start_indx)
        for i in range(start_indx, num_classes+start_indx):
            PATH = os.path.join(self.root, self.directories[i])
            if os.path.isdir(PATH):
                for file in os.listdir(PATH):
                    if file.endswith(self.img_type):
                        self.annotations.append(os.path.join(PATH, file))
                        self.class_labels.append(class_label)
                class_label +=1
        print(len(self.annotations))
        print("done!")
        
    def __getitem__(self, index):
        img_id = self.annotations[index]
        label  = self.class_labels[index]
        label = torch.Tensor([label]).type(torch.LongTensor)
        
        img = Image.open(img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        if self.ret_class:
            return img, label, index
        else:
            return img

    def get_all_labels(self):
        return list(map(int, self.class_labels))

    def __len__(self):
        return len(self.annotations)
