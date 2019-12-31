import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        #self.hidden2idx = nn.Linear(hidden_dim,
        self.linear = nn.Linear(hidden_size, vocab_size) 
        
        #self.hidden = self.init_hidden()
        
#     def init_hidden(self):
        
#         return (torch.zeros(1,1,self.hidden_size()),
#                 torch.zeros(1,1,self.hidden_size()))
    
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        captions = captions[:, :-1]
        captions = self.embed(captions)
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        lstm_out, _ = self.lstm(inputs, None)
        outputs = self.linear(lstm_out)
        return outputs
        
        
                

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        lstm_state = None
        for i in range(max_len):
            lstm_out, lstm_state = self.lstm(inputs, lstm_state)
            output = self.linear(lstm_out) 
            
            prediction = torch.argmax(output, dim = 2)
            predicted_index = prediction.item() 
            sentence.append(predicted_index) 
            
            if predicted_index ==1:
                break
            inputs = self.embed(prediction)
            
        return sentence
                