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
import LSTM
import NNLM

def parse_args():
    parse_args = argparse.ArgumentParser(description='Take args for data')
    parse_args.add_argument('--model' , type=str, default='lstm', help='Model to use')
    parse_args.add_argument('--test' , action='store_true', help='Test model')
    parse_args.add_argument('--model_path' , type=str, default='../data/models/LSTM.pt', help='Model path')
    parse_args.add_argument('--data_path' , type=str, default='../data/lm_dataloader.pkl', help='Data path')
    return parse_args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    glove_embeddings = pickle.load(open('../data/embeddings.pkl','rb'))
    if args.model == 'lstm':
        model = LSTM.LSTM_LM(glove_embeddings, hp.HIDDEN_LAYER, hp.NUM_LAYERS, hp.DROPOUT).to(device)
        model.load_state_dict(torch.load('../data/models/LSTM.pt'))
    else:
        model = NNLM.NNLM(glove_embeddings, hp.HIDDEN_LAYER,hp.N_GRAM, hp.DROPOUT).to(device)
        model.load_state_dict(torch.load('../data/models/NNLM.pt'))
    word_to_id = json.load(open('../data/word_to_id.json','r'))
    model.eval()
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("Model Loaded")
    if args.test: 
        
        if args.model == 'lstm':
            print("LSTM")
            loader__= pickle.load(open(args.data_path,'rb'))
            loss, perp = LSTM.test(model, loader__['test'], criterion,device)
            print("Loss: ", loss)
            print("Perplexity: ", perp)
            
        else:
            print("NNLM")
            print(args.data_path)
            loader__= pickle.load(open(args.data_path,'rb'))
            loss ,perp= NNLM.test_model(model, loader__['test'], criterion, device)
            print("Loss: ", loss)
            print("Perplexity: ", perp)
    else:
        if args.model == 'lstm':
            model.eval()
            print("Model Loaded")
            while(True):
                print("Enter the starting word")
                word = input()
                print("Enter the number of words to generate")
                num_words = int(input())
                generated_text = LSTM.LSTM_LM.generate_text(model , word_to_id, word, num_words)
                print(generated_text)
        else : 
            print("not yet implemented")
        
    
            

    

        
    
    
