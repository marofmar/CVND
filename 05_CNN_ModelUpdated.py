"""
CVND 05
- Wed 6 Nov 2019
- Updated CNN model.py
- I kept facing “size mismatch” error while training the model on sample dataset and FIXED IT!
- What I learned is, when we face “m1[a*b] and m2[c*d]” not match, 
  we should only take care of making the b and c get equal value! 
  (m1[batchsize*in_features], m2[in_features, out_features])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as I
'''
w: width of an image(assuming square sized input)
f: filter size (f*f, square)
s: stride
p: padding
'''
def calc(w,f,s):
    return (w-f)/s +1 




class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

       
        
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 5),#ch, out, kernel
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2)) # kernel_size, stride, padding
                                                     # 14
        
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 5),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2)) # 60 
        
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,1)) # 62
        
        self.layer4 = nn.Sequential(nn.Conv2d(64, 32, 3),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))  # 30   
        
        self.layer5 = nn.Sequential(nn.Conv2d(32, 16, 3),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2)) # 14   
                                                    
        self.fc1 = nn.Linear(16*11*11,64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,136)
        

        
        

        
    def forward(self, x):
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = self.layer5(x)
        print(x.shape)
  

        x = x.view(-1,16*11*11,) #input 61952
    # why not x = x.view(-1, x.size(0))?

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
        
