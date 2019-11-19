'''
CVND 17/30 (Tue 19 Nov 2019) #30DaysofUdacity #CVND #PracticeMakesPerfect
I am still working on Speech Tagging part using LSTM algorithm. 
I thought I knew about this, but it turned out I have missed and skipped many important parts in it. 
Also, reviewed about class structure in Python language. 
'''

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        '''initialize the layers of this model'''
        super.hidden_dim = hidden_dim 

        #embedding layer that runs words into a vector of a specified size 
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim) #using embedding function, vocab size dim reduced into embedding size dim

        # the LSTM takes embedding word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim) # using LSTM layer input number is 'embedding_dim' output number is 'hidden_dim'

        # the linear layer that maps the hidden state output dimension
        # to the number of tags we want as output, tagset_size ( in this case thi sis 3 tags) 
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size) # sending hidden_dim number of hidden layer outputs to final output, tagset 

        # initialize the hidden state( see code below) 
        self.hidden = self.init_hidden() 

    def init_hidden(self):
        '''
        At the start of training, we need to initialize a hidden state: 
        there will be none becuase the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeros and of a specified size. 
        '''
        # The axes dimensions are (n_layers, batch_size, hidden_dim) 
        return (torch.zeros(1,1,self.hidden_dim),
                torch.zeros(1,1,self.hidden_dim)) 
        
    def forward(self, sentence):
        '''Define the feedforward behavior of the model '''
        # create embdded word vectors for each word in a sentence 
        embeds = self.word_embeddings(sentence) 

        # get the output and hidden state by passing the lstm over our word embeddings 
        # the lstm takes in our embeddings and hidden state 
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden) 

        tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1)) 
        tag_scores = F.log_softmax(tag_outputs, dim = 1) 

        return tag_scores 
