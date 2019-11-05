"""
Tue 5 Nov 2019
CNN to detect facial keypoints

Things have updated

  1. x.view(@@@, -1) to x.view(-1,@@@) 
  2. before two conv layers to five layers
  3. before single fully connected layer to three fully connected ones

Hope this new architecture works better than before! 

"""




import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
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

        #*****************************************************
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 5),#ch, out, kernel
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2)) # kernel_size, stride, padding
                                                     # 28
        
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
                                                    
        self.fc1 = nn.Linear(16*14*14,64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,136)
        

        
        

        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(-1, 16*14*14) 

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
