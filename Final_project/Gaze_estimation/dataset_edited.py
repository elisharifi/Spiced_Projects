import os
import numpy as np
import cv2


import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


class Mpiigaze(Dataset): 
  def __init__(self, pathorg, root, transform, train, angle,fold=0):
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    path=pathorg.copy()
    if train==True:
      path.pop(fold)
    else:
      path=path[fold]
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                ####
                # import pdb
                # pdb.set_trace() 
                ###
                print(i)
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 42 and abs((label[1]*180/np.pi)) <= 42:
                self.lines.append(line)
   
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))
    ###############
    # remove lines without images
    keep_ids = []
    for i, l in enumerate(self.lines):
      line = l.strip().split(" ")
      face = line[0]
      if os.path.exists(os.path.join(self.root, face)):
        keep_ids.append(i)

    # import pdb
    # pdb.set_trace()
    print(f"Number of removed frames: {len(self.lines) - len(keep_ids)}")  
    self.lines = [self.lines[i] for i in keep_ids]

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)


    pitch = label[0]* 180 / np.pi
    yaw = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))

    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)
    
    if self.transform:
        img = self.transform(img)        
    
    # Bin values
    bins = np.array(range(-42, 42,3))
    binned_pose = np.digitize([pitch, yaw], bins) - 1

    labels = binned_pose
    cont_labels = torch.FloatTensor([pitch, yaw])


    return img, labels, cont_labels, name



