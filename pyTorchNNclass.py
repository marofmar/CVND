import torch.nn as nn 
import torch.nn.functional as F


class Net(nn.Moduel):
  def __init__(self, n_classes):
    super(Net, self).__init__() 
    self.conv1 = nn.Conv2d(1, 32, 5)
    # 1 channel, grayscale image
    # 32 output channels/feature maps 
    # 5x5 convolutional kernel
    
    self.pool = nn.MaxPool2d(2,2)
    # pooling kernel size 2x2
    # stride 2
    
    self.fc1 = nn.Linear(32*4, n_classes) 
    # fully connected layer
    # 32*4 input size to account for the downsampled img size after pooling
    # num_classes, ouput
    
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x))
    # one (conv+relu) + pool layers
    
    x = x.view(x.size(0), -1) 
    # prep for the linear layer by flattening the feature maps into feature vectors 
    
    x = F.relu(self.fc1(x)) 
    
    return x 
      
