# importing the libraries 
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import random
from torch import cuda
import config_hp as hp
from pprint import pprint
import pickle 
from  data_maker import *
WANDB_SILENT = "true"
import wandb
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def parse_args():
    parser = argparse.ArgumentParser(description='Take args for data')
    parser.add_argument('--wandb' , action='store_true', help='Use wandb')
    parser.add_argument('--save' , action='store_true', help='Save model')
    return parser.parse_args()

class LSTM_LM(nn.Module):
    def __init__(self, glove_embeddings,hidden_layer, num_layers, dropout):
        super(LSTM_LM, self).__init__()
        self.vocab_size = glove_embeddings.shape[0]
        self.embedding_dim = glove_embeddings.shape[1]
        self.hidden_layer = hidden_layer
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(glove_embeddings)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_layer , num_layers=self.num_layers, dropout=dropout, batch_first = True)
        self.fc = nn.Linear(hidden_layer,self.vocab_size )
        self.dropout = nn.Dropout(dropout)
        
       
    def forward(self, x, hidden=None):
        # print(x.shape)
       
        embeds = self.embedding(x) # batch_size, seq_len, embedding_dim
        # print(embeds.shape)
        if hidden is None: 
            lstm_out, hidden = self.lstm(embeds) # batch_size, seq_len, hidden_layer
        else:# useful for generation
            lstm_out, hidden = self.lstm(embeds, hidden)
       
        lstm_out = self.dropout(lstm_out) # batch_size, seq_len, hidden_layer
        out = self.fc(lstm_out) # batch_size, seq_len, vocab_size
       
        return out, hidden
    
    def generate_text(model, word_to_id, word, num_words):
        '''
        This function generates text using the trained model
        args: word_to_id: dict
              word: str
              num_words: int
        return: generated_text: str
        '''
        start_ = "<SOS>"
        start_id = word_to_id[start_]
        id_to_word = {v:k for k,v in word_to_id.items()}
        model.eval()
        generated_text = word + ' '
        if word not in word_to_id.keys():
            print("hi")
            word = "<OOV>"
        word_id = word_to_id[word]
        
        word_id = torch.tensor(word_id).view(1,1).to(device)
        hidden = None
        for i in range(num_words):
            output, hidden = model(word_id, hidden)
            output = output.view(-1)
            word_id = torch.argmax(output).view(1,1)
            generated_text += id_to_word[word_id.item()] + ' '
        return generated_text

def train(model, loader, optimizer, criterion, scheduler , wandb_, save_):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(hp.EPOCHS):
        total_loss = 0
        model.train()
        time_start = time.time()
        print('Epoch {}/{}'.format(epoch+1, hp.EPOCHS ))
        print('-' * 10)
        for batch_idx, (data, target) in enumerate(loader['train']):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, hidden = model(data)
            output = output.view(-1, output.shape[2])
            loss = criterion(output, target.view(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        total_loss /= len(loader['train'])
        perplexity = torch.exp(torch.tensor(total_loss))
        print('Training Loss: {:.4f}'.format(total_loss))
        print('Training Perplexity: {:.4f}'.format(perplexity))
        if wandb_:
            wandb.log({"Train Loss": total_loss, "Train Perplexity": perplexity})
        val_loss, val_perplexity = test(model, loader['val'], criterion)
        print('Validation Loss: {:.4f}'.format(val_loss))
        print('Validation Perplexity: {:.4f}'.format(val_perplexity))
        if wandb_:
            wandb.log({"Validation Loss": val_loss, "Validation Perplexity": val_perplexity})
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_:
                torch.save(model.state_dict(), '../data/models/LSTM.pt')
        print('Time taken for epoch: {:.4f}'.format(time.time()-time_start))
        print('-' * 10)
    return model

def test(model, loader, criterion, device= 'cuda'):
    model.eval()
    model.to(device)
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output, hidden = model(data)
            output = output.view(-1, output.shape[2])
            loss = criterion(output, target.view(-1))
            total_loss += loss.item()
    total_loss /= len(loader)
    perplexity = torch.exp(torch.tensor(total_loss))
    return total_loss, perplexity

if __name__ == "__main__":
    args = parse_args()
    wandb_to = args.wandb
    save_to = args.save
    if wandb_to:
        wandb.init(project="LSTM-anlp", config = {
            "epochs": hp.EPOCHS,
            "batch_size": hp.BATCH_SIZE,
            "learning_rate": hp.LEARNING_RATE,
            "hidden_layer": hp.HIDDEN_LAYER,
            "num_layers": hp.NUM_LAYERS,
            "dropout": hp.DROPOUT,
            "n_gram": hp.N_GRAM,
            "threshold": hp.THRESHOLD,
            "gamma": hp.GAMMA,
            "step_size": hp.STEP_SIZE,
        }
        )
        
    print("Loading data...")
    with open('../data/embeddings.pkl', 'rb') as f:
        glove_embeddings = pickle.load(f)
    with open('../data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('../data/lm_dataloader.pkl','rb') as f:
        loaders = pickle.load(f)
    print("Data loaded!")
    model = LSTM_LM(glove_embeddings, hp.HIDDEN_LAYER, hp.NUM_LAYERS, hp.DROPOUT)
    criterion = nn.CrossEntropyLoss( ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=hp.GAMMA, patience=hp.STEP_SIZE, verbose=True)
    model = train(model, loaders, optimizer, criterion, scheduler, wandb_to, save_to)
    test_loss, test_perplexity = test(model, loaders['test'], criterion)
    print('Test Loss: {:.4f}'.format(test_loss))
    print('Test Perplexity: {:.4f}'.format(test_perplexity))