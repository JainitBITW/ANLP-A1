import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import nltk 
nltk.download('punkt')
import argparse 
from nltk.corpus import stopwords
import random
from torch import cuda
import config_hp as hp
from pprint import pprint
import pickle 
from  data_maker import *
import wandb
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def parse_args():
    parser = argparse.ArgumentParser(description='Take args for data')
    parser.add_argument('--wandb' , action='store_true', help='Use wandb')
    parser.add_argument('--save' , action='store_true', help='Save model')
    return parser.parse_args()
    

class NNLM(nn.Module):
    """This class implements the Neural Network Language Model"""
    def __init__( self, glove_embeddings, hidden_layer,  n_gram, drpout= 0.5):
        super(NNLM, self).__init__()
        self.vocab_size = glove_embeddings.shape[0]
        self.embeddings_dim = glove_embeddings.shape[1]
        print(self.embeddings_dim)
        self.embeddings = nn.Embedding.from_pretrained(glove_embeddings)
        self.linear1 = nn.Linear(( self.embeddings_dim)*n_gram, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, self.vocab_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=drpout)
        self.n_gram = n_gram
        

    
    def forward(self, x):
        x = self.embeddings(x) # (batch_size, n_gram, embedding_dim)
        # print(x.shape)
        x = x.view(x.shape[0], -1) # (batch_size, n_gram*embedding_dim)
        # print(x.shape)
        x = self.linear1(x) # (batch_size, hidden_layer)
        x = self.relu1(x) # (batch_size, hidden_layer)
        x = self.dropout1(x) # (batch_size, hidden_layer) 
        x = self.linear2(x) # (batch_size, vocab_size)
       
        return x

class NNLM_without_pretrained(nn.Module):
    """This class implements the Neural Network Language Model"""
    def __init__( self, glove_embeddings, hidden_layer,  n_gram, drpout= 0.5):
        super(NNLM_without_pretrained, self).__init__()
        self.vocab_size = glove_embeddings.shape[0]
        self.embeddings_dim = glove_embeddings.shape[1]
        print(self.embeddings_dim)
        self.embeddings = nn.Embedding(self.vocab_size, self.embeddings_dim)
        self.linear1 = nn.Linear(( self.embeddings_dim)*n_gram, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, self.vocab_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=drpout)
        self.n_gram = n_gram
        

    
    def forward(self, x):
        x = self.embeddings(x) # (batch_size, n_gram, embedding_dim)
        # print(x.shape)
        x = x.view(x.shape[0], -1) # (batch_size, n_gram*embedding_dim)
        # print(x.shape)
        x = self.linear1(x) # (batch_size, hidden_layer)
        x = self.relu1(x) # (batch_size, hidden_layer)
        x = self.dropout1(x) # (batch_size, hidden_layer) 
        x = self.linear2(x) # (batch_size, vocab_size)
        return x
    
def train_model(model, dataloader, optimizer, criterion, scheduler,wandb_to, to_save):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(hp.EPOCHS):
        timee_now = time.time()
        print('Epoch {}/{}'.format(epoch+1, hp.EPOCHS ))
        print('-' * 10)
        total_loss = 0
        for x,y in dataloader['train']:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Train Loss: {total_loss/len(dataloader["train"])}')
        perplexity = torch.exp(torch.tensor(total_loss/len(dataloader["train"])))
        print(f'Train Perplexity: {perplexity}')
        val_loss, val_perplexity = test_model(model, dataloader['val'], criterion)
        print(f'Val Loss: {val_loss}')
        print(f'Val Perplexity: {val_perplexity}')
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if to_save:
                torch.save(model.state_dict(), '../data/models/NNLM.pt')
        if wandb_to:
            wandb.log({"Train Loss": loss/len(dataloader["train"]), "Train Perplexity": perplexity, "Val Loss": val_loss, "Val Perplexity": val_perplexity})
        print(f'Time taken for epoch: {time.time()-timee_now}')
        print('-' * 10)
    return model

def test_model(model, dataloader, criterion ,  device= 'cuda'):
    model.eval()
    model.to(device)
    total_loss = 0
    with torch.no_grad():
        for x , y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
    
    total_loss = total_loss/len(dataloader)
    perplexity = torch.exp(torch.tensor(total_loss))
            
    return total_loss, perplexity



if __name__ == '__main__': 
    with open('../data/embeddings.pkl', 'rb') as f:
        glove_embeddings = pickle.load(f)
    with open('../data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('../data/ngrams_dataloader.pkl', 'rb') as f:
        ngrams_dataloader = pickle.load(f)
    wandb_to = parse_args().wandb
    if wandb_to: # if wandb is used
        wandb.init(project="anlp-a1", config = {
            "batch_size": hp.BATCH_SIZE,
            "epochs": hp.EPOCHS,
            "hidden_layer": hp.HIDDEN_LAYER,
            "n_gram": hp.N_GRAM,
            "threshold": hp.THRESHOLD,
            "gamma": hp.GAMMA,
            "learning_rate": hp.LEARNING_RATE,
            "step_size": hp.STEP_SIZE,
            "dropout": hp.DROPOUT,
            "embeddings_dim": glove_embeddings.shape[1],
            "vocab_size": glove_embeddings.shape[0]
        })
      
    # print(glove_embeddings.shape)
    # print(vocab)
    initial_lr = hp.LEARNING_RATE
    model = NNLM(glove_embeddings, hp.HIDDEN_LAYER, hp.N_GRAM, hp.DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=hp.STEP_SIZE, factor=hp.GAMMA, verbose=True)
    criterion = nn.CrossEntropyLoss()
    model = train_model(model, ngrams_dataloader, optimizer, criterion, scheduler,  wandb_to=wandb_to , to_save = parse_args().save)
    loss , perp = test_model(model, ngrams_dataloader['test'], criterion)
    print(f'Test Loss: {loss}')
    print(f'Test Perplexity: {perp}')
